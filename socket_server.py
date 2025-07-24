import json
import os
import socket
import struct
import threading
import uuid
from collections import OrderedDict
from typing import Dict

import numpy as np
# TODO：参考legged_gym/legged_gym/scripts/play.py中对环境的使用，替换libero并封装为issac gym函数形式

from legged_gym.envs import *
from legged_gym.utils import  get_args,  task_registry
from terrain_base.config import terrain_config

import torch
import faulthandler

from libero.libero import get_libero_path
from libero.libero.benchmark import Benchmark, get_benchmark
from libero.libero.envs import OffScreenRenderEnv


class Session:
    def __init__(self, benchmark: Benchmark, task_id: int, env_kwargs):
        self.benchmark = benchmark
        self.task_id = task_id
        self.env_kwargs = env_kwargs

        #TODO 这里的train_cfg是否可用于test阶段
        env_cfg, train_cfg = task_registry.get_cfgs(name=env_kwargs.task)
        # override some parameters for testing
        if env_kwargs.nodelay:
            env_cfg.domain_rand.action_delay_view = 0

        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 1000
        env_cfg.commands.resampling_time = 60
        env_cfg.rewards.is_play = False

        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 10
        env_cfg.terrain.max_init_terrain_level = 1

        env_cfg.terrain.height = [0.01, 0.02]

        env_cfg.depth.angle = [0, 1]
        env_cfg.noise.add_noise = True
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.push_interval_s = 8
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False

        depth_latent_buffer = []
        # prepare environment
        env: HumanoidRobot
        self.env, _ = task_registry.make_env(name=env_kwargs.task, args=env_kwargs, env_cfg=env_cfg)


class LiberoServer:
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        self.sessions: Dict[str, Session] = {}
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
    def start(self):
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")
        
        try:
            while True:
                client_socket, addr = self.socket.accept()
                print(f"Connection from {addr}")
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, addr)
                )
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            self.socket.close()
            # Clean up any remaining sessions
            for session_id, session in list(self.sessions.items()):
                session.env.close()
            self.sessions.clear()

    def receive_data(self, client_socket):
        # First receive the message size (4 bytes)
        size_data = client_socket.recv(4)
        if not size_data:
            return None
            
        msg_size = struct.unpack('!I', size_data)[0]
        
        # Now receive the actual message data
        received_data = b''
        remaining = msg_size
        
        while remaining > 0:
            chunk = client_socket.recv(min(4096, remaining))
            if not chunk:
                break
            received_data += chunk
            remaining -= len(chunk)
            
        return json.loads(received_data.decode('utf-8'))

    def send_data(self, client_socket, data):
        # Serialize data to JSON and encode
        json_data = json.dumps(data).encode('utf-8')
        
        # Send message size first, then the message
        msg_size = struct.pack('!I', len(json_data))
        client_socket.sendall(msg_size + json_data)

    def handle_client(self, client_socket, addr):
        
        # 60s timeout
        client_socket.settimeout(60.0) 

        session_id = None
        try:
            while True:
                data = self.receive_data(client_socket)
                if data is None:
                    break
                    
                command = data.get('command')
                if not command:
                    self.send_data(client_socket, {"error": "No command specified"})
                    continue
                
                # Process different commands
                if command == 'create':
                    response = self.handle_create(data, addr[0])
                    session_id = response.get('session_id')
                elif command == 'reset':
                    session_id = data.get('session_id')
                    response = self.handle_reset(session_id)
                elif command == 'step':
                    session_id = data.get('session_id')
                    action = data.get('action')
                    response = self.handle_step(session_id, action)
                elif command == 'close':
                    session_id = data.get('session_id')
                    response = self.handle_close(session_id)
                else:
                    response = {"error": f"Unknown command: {command}"}
                
                self.send_data(client_socket, response)
                
        except Exception as e:
            print(f"Error handling client {addr}: {str(e)}")
        finally:
            # Clean up if client disconnects unexpectedly
            if session_id and session_id in self.sessions:
                try:
                    session = self.sessions.pop(session_id)
                    session.env.close()
                    print(f"Session {session_id} closed due to client disconnect")
                except:
                    pass
            client_socket.close()

    def jsonify_observation(self, obs):
        res = OrderedDict()
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                res[k] = v.tolist() if v.ndim > 0 else v.item()
            elif isinstance(v, np.bool_):
                res[k] = bool(v)
            else:
                res[k] = v
        return res

    def handle_create(self, data, client_ip):
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Parse the request data
        benchmark_name = data.get('benchmark_name', "libero_spatial")
        if benchmark_name not in ["libero_spatial", "libero_10", "libero_object", "libero_goal"]:
            return {"error": "Unsupported benchmark"}
            
        task_id = data.get('task_id')
        if task_id is None:
            return {"error": "task_id is required"}
            
        camera_width = data.get('camera_width', 128)
        camera_height = data.get('camera_height', 128)
        if camera_width > 512 or camera_height > 512:
            return {"error": "camera_width and camera_height must be <= 512"}
        
        benchmark: Benchmark = get_benchmark(benchmark_name)()
        task = benchmark.get_task(task_id) 

        # Prepare environment arguments
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": camera_height,
            "camera_widths": camera_width,
        }
        
        self.sessions[session_id] = Session(benchmark, task_id, env_args)
        
        print(f"Session created for {client_ip} with session ID {session_id}")
        return {"session_id": session_id, "task_name": task.name}

    def handle_close(self, session_id):
        if session_id and session_id in self.sessions:
            session = self.sessions.pop(session_id)
            session.env.close()
            print(f"Session {session_id} closed")
            return {"message": "Session closed"}
        else:
            return {"error": "Session not found"}
    
    def handle_reset(self, session_id):
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            return {"error": "Session not found"}
        
        benchmark = session.benchmark
        task_id = session.task_id
        task = benchmark.get_task(task_id)
        env = session.env

        env.reset()
        # Set seed for reproducibility
        env.seed(0)

        init_states = benchmark.get_task_init_states(task_id)
        obs = env.set_init_state(init_states[0])
        description = task.language

        return {
            "observation": self.jsonify_observation(obs),
            "description": description,
        }

    def handle_step(self, session_id, action):
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            return {"error": "Session not found"}
        
        env = session.env
        
        if action is None:
            return {"error": "action is required"}
        
        try:
            action = np.array(action)
            if action.ndim != 1 or action.shape[0] != 7:
                return {"error": "action must be a 1D array of shape (7,)"}
                
            observation, reward, done, info = env.step(action)

            observation = self.jsonify_observation(observation)
            reward = float(reward)
            done = bool(done)
            if isinstance(info, dict):
                info = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}
            
            return {
                "observation": observation,
                "reward": reward,
                "done": done,
                "info": info
            }
        except Exception as e:
            return {"error": f"Error during step: {str(e)}"}


if __name__ == '__main__':
    server = LiberoServer()
    server.start()