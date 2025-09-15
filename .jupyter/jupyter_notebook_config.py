# Configuración simple para evitar conflictos con OneDrive
c = get_config()

# Configuración básica del servidor
c.ServerApp.ip = '127.0.0.1'
c.ServerApp.port = 8888
c.ServerApp.open_browser = True
c.ServerApp.allow_root = False
c.ServerApp.allow_origin = '*'

# Configuración de seguridad
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = False

# Configuración de directorios (usar rutas absolutas)
import os
c.ServerApp.notebook_dir = os.path.abspath('notebooks')
c.ServerApp.root_dir = os.path.abspath('.')

# Configuración de kernels
c.MultiKernelManager.default_kernel_name = 'python3'

# Configuración de logging (solo errores importantes)
c.ServerApp.log_level = 'ERROR'
c.Application.log_level = 'ERROR'

# Evitar problemas con OneDrive
c.ServerApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors 'self' http://localhost:* https://localhost:*;"
    }
}

