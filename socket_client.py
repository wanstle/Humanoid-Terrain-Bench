import json
import os
import socket
import struct
import time
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np


def dejsonify_observation(obs: dict):
    ret = OrderedDict()
    for k, v in obs.items():
        ret[k] = np.array(v) if "image" in k else v
    return ret


class RemoteEnv:
    """
    Remote environment client that communicates with a LIBERO server using socket communication.
    Implements the same basic interface as gym.Env for easy integration.
    """
    
    def __init__(self, 
                 server_ip: str, 
                 server_port: int,
                 benchmark_name: str = "libero_spatial",
                 task_id: int = 0,
                 camera_width: int = 128,
                 camera_height: int = 128,
                 timeout: float = 30.0):
        """
        Initialize a connection to a remote LIBERO environment server.
        
        Args:
            server_ip: IP address of the server
            server_port: Port number of the server
            benchmark_name: Name of the benchmark to use ('libero_spatial', 'libero_10', etc.)
            task_id: Task ID to load
            camera_width: Width of camera images
            camera_height: Height of camera images
            timeout: Connection timeout in seconds
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.timeout = timeout
        
        self.socket = None
        self.session_id = None
        self.description = None
        
        # Connect to server and create session
        self._connect()
        self._create_session()
    
    def _connect(self):
        """Establish a socket connection to the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.server_ip, self.server_port))
        except Exception as e:
            raise ConnectionError(f"Failed to connect to server at {self.server_ip}:{self.server_port}. Error: {str(e)}")
    
    def _send_data(self, data: Dict):
        """Send data to the server."""
        json_data = json.dumps(data).encode('utf-8')
        msg_size = struct.pack('!I', len(json_data))
        self.socket.sendall(msg_size + json_data)
    
    def _receive_data(self) -> Dict:
        """Receive data from the server."""
        # First receive the message size (4 bytes)
        size_data = self.socket.recv(4)
        if not size_data:
            raise ConnectionError("Connection closed by server while receiving data size")
            
        msg_size = struct.unpack('!I', size_data)[0]
        
        # Receive the actual message data
        received_data = b''
        remaining = msg_size
        
        while remaining > 0:
            chunk = self.socket.recv(min(4096, remaining))
            if not chunk:
                raise ConnectionError("Connection closed by server while receiving data")
            received_data += chunk
            remaining -= len(chunk)
            
        return json.loads(received_data.decode('utf-8'))
    
    def _create_session(self):
        """Create a new session on the server."""
        request = {
            "command": "create",
            "benchmark_name": self.benchmark_name,
            "task_id": self.task_id,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height
        }
        
        self._send_data(request)
        response = self._receive_data()
        
        if "error" in response:
            raise RuntimeError(f"Failed to create session: {response['error']}")
            
        self.session_id = response["session_id"]
        self.task_name = response.get("task_name")
        print(f"Created session with ID: {self.session_id}")
    
    def reset(self):
        """Reset the environment and return initial observation."""
        if not self.session_id:
            raise RuntimeError("No active session. Create a new RemoteEnv instance.")
            
        request = {
            "command": "reset",
            "session_id": self.session_id
        }
        
        self._send_data(request)
        response = self._receive_data()
        
        if "error" in response:
            raise RuntimeError(f"Failed to reset environment: {response['error']}")
            
        self.description = response.get("description")
        return dejsonify_observation(response["observation"])
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Robot action as numpy array of shape (7,)
            
        Returns:
            observation, reward, done, info
        """
        if not self.session_id:
            raise RuntimeError("No active session. Create a new RemoteEnv instance.")
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(action, np.ndarray):
            action = action.tolist()
            
        request = {
            "command": "step",
            "session_id": self.session_id,
            "action": action
        }
        
        self._send_data(request)
        response = self._receive_data()
        
        if "error" in response:
            raise RuntimeError(f"Failed to step environment: {response['error']}")
            
        return (
            dejsonify_observation(response["observation"]),
            response["reward"],
            response["done"],
            response["info"]
        )
    
    def close(self):
        """Close the session and connection."""
        if self.session_id:
            try:
                request = {
                    "command": "close",
                    "session_id": self.session_id
                }
                self._send_data(request)
                response = self._receive_data()
                print(f"Closed session: {self.session_id}")
            except:
                print(f"Failed to close session {self.session_id}. It may have already been closed.")
                
            self.session_id = None
            
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        self.close()


def main():
    import cv2

    env = RemoteEnv(server_ip="localhost", server_port=9999)
    obs = env.reset()
   
    cv2.imshow(
        "Observation",
        obs["agentview_image"].astype(np.uint8)[::-1]
    )
    cv2.waitKey(0)

    done = False
    while not done:
        # action = np.zeros(7)  # Replace with your policy
        action = np.zeros(7)
        action[:3] = np.random.uniform(-1, 1, size=3)  # Random action for testing
        obs, reward, done, info = env.step(action)

        cv2.imshow(
            "Observation",
            obs["agentview_image"].astype(np.uint8)[::-1]
        )
        cv2.waitKey(100)

    env.close()


if __name__ == "__main__":
    main()