from redis import Redis
from rq import Worker, Queue, Connection

listen = ["file_tasks"]
redis_conn = Redis(host="localhost", port=6380, db=0)

if __name__ == "__main__":
    with Connection(redis_conn):
        Worker(map(Queue, listen)).work()