# -----------------------------------------------------------------------------
# socket_server_client.py
# -----------------------------------------------------------------------------
# This single file contains **two** standalone Python entry‑points:
#   1. `socket_server.py`  – run with:  python socket_server_client.py --mode server
#   2. `client.py`         – run with:  python socket_server_client.py --mode client
# They communicate via a very small binary‑safe protocol:
#     [uint32 payload_len][payload_len bytes of UTF‑8 JSON]
# Each JSON message has at minimum the key "cmd" whose value determines the
# message type ("create_env", "reset", "step", "close").  The server replies to
# every request with a JSON message that always contains a field "ok" (bool).
# -----------------------------------------------------------------------------
# Server responsibilities
# 1) Accept a TCP connection and create an RL environment on a "create_env"
#    request (we use legged_gym's task_registry to stay consistent with your
#    play.py/humanoid_robot/base_task stack).
# 2) Service subsequent "reset" and "step" requests, delegating directly to the
#    environment.  Observations, rewards, done flags and infos are streamed
#    back as JSON (with small numpy arrays serialised as lists to keep things
#    simple and language‑agnostic).
# 3) Record episode metrics (cumulative reward, episode length, wall‑clock).
# 4) Before closing the connection, send the aggregated evaluation statistics to
#    the client.
# -----------------------------------------------------------------------------
# NOTE  ➜  Only **one** environment per connection is supported to keep the
#          reference implementation concise.  Adding gym.vectorised support is a
#          straightforward extension.
# -----------------------------------------------------------------------------
# Dependencies (all standard except legged_gym):
#   pip install legged_gym==0.1 torch numpy
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# -----------------------------------------------------------------------------
# ----- Utility functions ------------------------------------------------------
# -----------------------------------------------------------------------------

UINT32_STRUCT = struct.Struct("!I")  # 4‑byte network‑order unsigned int


def _send(sock: socket.socket, payload: Dict[str, Any]) -> None:
    """Send a dict as length‑prefixed JSON."""
    data = json.dumps(payload).encode()
    sock.sendall(UINT32_STRUCT.pack(len(data)) + data)


def _recv(sock: socket.socket) -> Dict[str, Any]:
    """Blocking receive of one length‑prefixed JSON packet."""
    # read 4‑byte length
    raw_len = _recvall(sock, UINT32_STRUCT.size)
    if not raw_len:
        raise ConnectionError("Socket closed by peer while reading length header")
    (length,) = UINT32_STRUCT.unpack(raw_len)
    # read payload
    raw_payload = _recvall(sock, length)
    if raw_payload is None:
        raise ConnectionError("Socket closed by peer while reading payload")
    return json.loads(raw_payload.decode())


def _recvall(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes or None if EOF is hit before all bytes arrive."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None  # EOF
        data.extend(packet)
    return bytes(data)


# -----------------------------------------------------------------------------
# ----- Server implementation --------------------------------------------------
# -----------------------------------------------------------------------------

class EvaluationServer:
    """Handle a single contestant connection."""

    def __init__(self, conn: socket.socket, addr: Tuple[str, int]):
        self.conn = conn
        self.addr = addr
        self.env = None  # will be created after "create_env"
        self.metrics: List[Dict[str, Any]] = []  # one dict per finished episode
        self.current_cumulative_reward: float = 0.0
        self.current_steps: int = 0
        self._episode_start_wall: float = 0.0

    # ------------------------------------------------------------------
    # Main per‑client loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        print(f"[SERVER] ▶ New connection from {self.addr}")
        try:
            while True:
                msg = _recv(self.conn)
                cmd = msg.get("cmd")
                if cmd == "create_env":
                    self._handle_create_env(msg)
                elif cmd == "reset":
                    self._handle_reset(msg)
                elif cmd == "step":
                    self._handle_step(msg)
                elif cmd == "close":
                    self._handle_close()
                    break
                else:
                    _send(self.conn, {"ok": False, "error": f"Unknown cmd: {cmd}"})
        except ConnectionError:
            print(f"[SERVER] ✖ Connection lost from {self.addr}")
        finally:
            self.conn.close()
            if self.env is not None:
                self.env.close()
            print(f"[SERVER] ◀ Connection closed {self.addr}")

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def _handle_create_env(self, msg: Dict[str, Any]):
        if self.env is not None:
            _send(self.conn, {"ok": False, "error": "env already created"})
            return
        task_name = msg.get("task", "humanoid_robot")  # default task
        seed = int(msg.get("seed", 42))

        # ---- build env ------------------------------------------------
        try:
            from legged_gym.utils import task_registry
            env_cfg, _ = task_registry.get_cfgs(name=task_name)
            env_cfg.env.num_envs = 1  # single‑env for eval simplicity
            env, _ = task_registry.make_env(name=task_name, args=SimpleNamespace(), env_cfg=env_cfg)
            env.seed(seed)
            self.env = env
        except Exception as exc:
            _send(self.conn, {"ok": False, "error": f"Failed to create env: {exc}"})
            return

        obs = self.env.reset()
        self.current_cumulative_reward = 0.0
        self.current_steps = 0
        self._episode_start_wall = time.time()

        _send(self.conn, {"ok": True, "obs": _to_jsonable(obs)})
        print(f"[SERVER]   Environment '{task_name}' created for {self.addr}")

    def _handle_reset(self, _msg: Dict[str, Any]):
        if self.env is None:
            _send(self.conn, {"ok": False, "error": "env not yet created"})
            return
        # Log previous episode if it existed
        if self.current_steps > 0:
            self._log_episode()
        obs = self.env.reset()
        self.current_cumulative_reward = 0.0
        self.current_steps = 0
        self._episode_start_wall = time.time()
        _send(self.conn, {"ok": True, "obs": _to_jsonable(obs)})

    def _handle_step(self, msg: Dict[str, Any]):
        if self.env is None:
            _send(self.conn, {"ok": False, "error": "env not yet created"})
            return
        action = np.asarray(msg.get("action"), dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]  # add batch dim
        obs, reward, done, info = self.env.step(torch.as_tensor(action, device=self.env.device))
        self.current_cumulative_reward += float(reward[0])
        self.current_steps += 1
        if bool(done[0]):
            self._log_episode()
        _send(
            self.conn,
            {
                "ok": True,
                "obs": _to_jsonable(obs),
                "reward": float(reward[0]),
                "done": bool(done[0]),
                "info": info[0] if isinstance(info, (list, tuple)) else info,
            },
        )

    def _handle_close(self):
        # Log final episode if still open
        if self.env is not None and self.current_steps > 0:
            self._log_episode()
        summary = _summarise_metrics(self.metrics)
        _send(self.conn, {"ok": True, "summary": summary})
        print(f"[SERVER]   Sent evaluation summary to {self.addr}: {summary}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log_episode(self):
        self.metrics.append(
            {
                "return": self.current_cumulative_reward,
                "length": self.current_steps,
                "wall_time": time.time() - self._episode_start_wall,
            }
        )
        print(
            f"[SERVER]   Episode finished – return={self.current_cumulative_reward:.2f} "
            f"steps={self.current_steps}"
        )
        self.current_cumulative_reward = 0.0
        self.current_steps = 0


# -----------------------------------------------------------------------------
# ----- Client implementation --------------------------------------------------
# -----------------------------------------------------------------------------

class DemoClient:
    """A minimal reference client that plays with random actions."""

    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port))

    def run(self):
        # 1) Create env
        _send(self.sock, {"cmd": "create_env", "task": "humanoid_robot", "seed": 123})
        resp = _recv(self.sock)
        assert resp["ok"], resp
        obs = np.array(resp["obs"], dtype=np.float32)
        print("[CLIENT] Environment created. obs shape=", obs.shape)

        # 2) Evaluate a fixed number of episodes with random actions
        rng = np.random.default_rng(0)
        n_episodes = 3
        for ep in range(n_episodes):
            _send(self.sock, {"cmd": "reset"})
            obs = np.asarray(_recv(self.sock)["obs"], dtype=np.float32)
            done = False
            while not done:
                action = rng.standard_normal(obs.shape[-1])  # very naive
                _send(self.sock, {"cmd": "step", "action": action.tolist()})
                resp = _recv(self.sock)
                obs = np.asarray(resp["obs"], dtype=np.float32)
                done = resp["done"]
        # 3) Close connection and receive summary
        _send(self.sock, {"cmd": "close"})
        summary = _recv(self.sock)
        print("[CLIENT] Evaluation summary:", summary["summary"])
        self.sock.close()


# -----------------------------------------------------------------------------
# ----- Metric helpers ---------------------------------------------------------
# -----------------------------------------------------------------------------

def _to_jsonable(x: Any) -> Any:
    """Convert numpy/tensor obs to python native lists for JSON serialisation."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x  # already json‑able


def _summarise_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    returns = [m["return"] for m in metrics]
    lengths = [m["length"] for m in metrics]
    wall_times = [m["wall_time"] for m in metrics]
    return {
        "episodes": len(metrics),
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "wall_time_total": float(np.sum(wall_times)),
    }


# -----------------------------------------------------------------------------
# ----- Main entry -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _run_server(port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", port))
        s.listen()
        print(f"[SERVER] 🚀 Listening on 0.0.0.0:{port}")
        while True:
            conn, addr = s.accept()
            handler = EvaluationServer(conn, addr)
            threading.Thread(target=handler.run, daemon=True).start()


def _run_client(host: str, port: int):
    DemoClient(host, port).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legged‑Gym evaluation socket server / client")
    parser.add_argument("--mode", choices=["server", "client"], default="client")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="server host (client mode)")
    args = parser.parse_args()

    if args.mode == "server":
        _run_server(args.port)
    else:
        _run_client(args.host, args.port)