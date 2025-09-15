# Configuración personalizada para Jupyter
c = get_config()

# Configuración del servidor
c.ServerApp.ip = '127.0.0.1'
c.ServerApp.port = 8888
c.ServerApp.open_browser = True
c.ServerApp.allow_root = True
c.ServerApp.allow_origin = '*'

# Configuración de seguridad
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = True

# Configuración de archivos
c.ServerApp.notebook_dir = 'notebooks'
c.ServerApp.root_dir = '.'

# Configuración de kernels
c.MultiKernelManager.default_kernel_name = 'python3'

# Configuración de logging
c.ServerApp.log_level = 'INFO'

