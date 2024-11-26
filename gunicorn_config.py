# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "0.0.0.0:8003"
backlog = 2048

# Worker processes
workers = 4
worker_class = 'sync'
worker_connections = 1000
timeout = 120
keepalive = 2

# Process naming
proc_name = 'translation-app'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None 