Una vez vistas todas las herramientas que rodean el uso de Magerit, podemos establecer un flujo de trabajo que permita organizar nuestro proyecto y asegurar su mantenibilidad a lo largo del tiempo.
Para este workflow mi entorno de trabajo es el siguiente:
- Ordenador personal con Ubuntu, sería equivalente WSL o una maquina virtual. 
- VSCode como editor de codigo
- Terminal. Yo usaré la terminal integrada, aunque se podría trabajar con la terminal integrada de VSCode

## 1. (Local) Preparación del venv

Lo primero que debemos hacer es **crear un venv con una versión especifica de Python**. 
Para ello podemos usar `pyenv`
#### Instalar la versión de Python
Si no tenemos instalada la versión de Python que queremos, lo hacemos. En este caso, usaremos `python 3.11.5`
```bash
rodrigo@PC:~/workflow$ pyenv install 3.11.5
Downloading Python-3.11.5.tar.xz...
-> https://www.python.org/ftp/python/3.11.5/Python-3.11.5.tar.xz
Installing Python-3.11.5...
patching file setup.py
Hunk #1 succeeded at 306 (offset -1 lines).
Installed Python-3.11.5 to /home/rodrigo/.pyenv/versions/3.11.5

```

Comprobamos que se haya instalado correctamente con `pyenv versions`
```bash
rodrigo@PC:~/workflow$ pyenv versions
* system (set by /home/rodrigo/.pyenv/version)
  3.6.15
  3.11.5
  3.11.15
  3.12.3
```

#### Activar la versión de Python en directorio de trabajo
Nos situamos en el directorio de trabajo (en mi caso `/workflow` ) y ejecutamos el mandato `pyenv local XX.XX.XX`. Este comando le dice al sistema que, en este directorio, y todos los que sigan su árbol, se usará una versión especifica de Python. Ignorando la versión de Python global que tengamos instalada.
```bash
rodrigo@PC:~/workflow$ pyenv local 3.11.5
#Comprobamos la versión que usa este directorio
rodrigo@PC:~/workflow$ python --version
Python 3.11.5
```

### Crear el entorno
Ahora creamos un venv con el nombre que queramos. Como ya hemos indicado la versión de Python que usaremos, podemos usar directamente la versión simplificada del mandato `python -m venv mi_entorno` 
```bash
rodrigo@PC:~/workflow$ python -m venv mi_entorno
```

### Activar el entorno
Activamos el entorno y comprobamos que versión de Python usa el entorno.
```bash
rodrigo@PC:~/workflow$ source mi_entorno/bin/activate
#Se activo el entorno. comprobamos la versión de Python que usa
(mi_entorno) rodrigo@PC:~/workflow$ python --version
Python 3.11.5
```

> [!warning] ADVERTENCIA
> Si no se indica la versión de Python que queremos usar (con `pyenv local XX.XX.XX`), el entorno virtual se creará con una versión de Python incorrecta.

> [!info] NOTA
> Esto solo se hace una vez por proyecto.
## 2. (Local) Desarrollar el código
Este paso es auto explicativo: Desarrolla el código en tu IDE preferido.

Aunque muchas veces no podamos correr el codigo por la cantidad de recursos que necesita, podemos seguir ciertos pasos para corroborar que el código no contiene errores comunes.
### Linters
Los Linters analizan el texto del código en busca de errores de sintaxis, variable sin usar e incluso malas practicas.
Los IDE modernos suelen tener uno integrado, pero podemos ejecutar uno nosotros mismos. En este ejemplo, usaremos `PyLint`
```bash
#Instalamos PyLint
(mi_entorno) rodrigo@PC:~/workflow$ pip install pylint
```

Podemos tener errores de sintaxis
```bash
(mi_entorno) rodrigo@PC:~/workflow$ pylint benchmark.py 
************* Module benchmark
benchmark.py:18:9: E0001: Parsing failed: 'invalid syntax (benchmark, line 18)' (syntax-error)
```

O si no los tenemos, revisa las buenas practicas y nos entrega una calificación.
```bash

(mi_entorno) rodrigo@PC:~/workflow$ pylint benchmark.py 
************* Module benchmark
benchmark.py:1:0: C0114: Missing module docstring (missing-module-docstring)
benchmark.py:7:0: C0116: Missing function or method docstring (missing-function-docstring)
benchmark.py:17:4: C0103: Variable name "camelCase" doesnt conform to snake_case naming style (invalid-name)
benchmark.py:17:4: W0612: Unused variable 'camelCase' (unused-variable)
benchmark.py:2:0: C0411: standard import "time" should be placed before third party import "torch" (wrong-import-order)

-----------------------------------
Your code has been rated at 8.08/10

(mi_entorno) rodrigo@PC:~/workflow$ 
```

### "Compilar" el codigo
Aunque Python sea un lenguaje interpretado, es decir, no tiene tipos en el sentido estricto de la palabra; podemos generar un archivo `.pyc (bytecode)` . Este nos avisará sobre errores de sintaxis.
El comando `python -m py_compyle codigo.py` viene por defecto, por lo que no hace falta instalar nada.
```bash
#Versión con errores
(mi_entorno) rodrigo@PC:~/workflow$ python -m py_compile benchmark.py 
  File "benchmark.py", line 17
    for :
        ^
SyntaxError: invalid syntax

#Versión correcta
(mi_entorno) rodrigo@PC:~/workflow$ python -m py_compile benchmark.py 

```

## 3. (Local) "Exportar" el entorno

### Exportar las dependencias
Exportamos las dependencias con el comando `pip freeze > requirements.txt`

```bash
(mi_entorno) rodrigo@PC:~/workflow$ pip freeze > requirements.txt
#Si queremos ver las dependencias
(mi_entorno) rodrigo@PC:~/workflow$ cat requirements.txt 
astroid==4.0.4
cuda-bindings==12.9.4
cuda-pathfinder==1.4.3
dill==0.4.1
filelock==3.25.2
fsspec==2026.2.0
isort==8.0.1
Jinja2==3.1.6
MarkupSafe==3.0.3
mccabe==0.7.0
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.3
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
packaging==26.0
pipdeptree==2.31.0
platformdirs==4.9.4
pylint==4.0.5
sympy==1.14.0
tomlkit==0.14.0
torch==2.10.0
triton==3.6.0
typing_extensions==4.15.0
```

## 4. (Local) Enviar los ficheros a Magerit
Una vez generada la lista de dependencias podemos enviar todos los ficheros que usaremos en Magerit. Generalmente enviarás tu codigo y sus requerimientos.
El envió se puede hacer a través del comando `scp fichero destino:ruta`

```bash
rodrigo@PC:~/workflow$ scp requirements.txt x123456@magerit.cesvima.upm.es:~
(x123456@magerit.cesvima.upm.es) Password: 
requirements.txt                                                                                                                                       100%  808    91.5KB/s   00:00
rodrigo@PC:~/workflow$ scp benchmark.py  x123456@magerit.cesvima.upm.es:~
(x12345@magerit.cesvima.upm.es) Password: 
benchmark.py                                                                                                                                           100% 1083   128.1KB/s   00:00   

```
La versión más rápida de este comando, envía los ficheros a la carpeta del usuario (representada por `~`)

# 5.(Magerit) Preparar el entorno y trabajar en Magerit
Una vez tenemos los ficheros en Magerit, podemos trabajar directamente allí. Podemos acceder usando el comando `ssh destino`
```bash
rodrigo@PC:~/workflow$ ssh x244645@magerit.cesvima.upm.es
(x244645@magerit.cesvima.upm.es) Password: 
```

### Estructura del proyecto
Una opción para organizar las carpetas del proyecto es la siguiente:
```bash 
PROJECT/experimento
	>	requirements.txt
	>	prepare_env.sh
	>	experimento.py
	>	exec_job.sh
	>	logs/

```

### División de trabajos
Para prevenir el uso innecesario de recursos del cluster y evitar repetición innecesaria de codigo, es una buena practica dividir la ejecución del trabajo en 2 partes: 
- Preparación del entorno virtual: Crear venv, instalar dependencias, descargar data sets
- Ejecución del trabajo: Ejecutar el programa `.py`

### 1. Preparación del entorno virtual
Se encarga de:
- Crear el entorno virtual, si no existe
- Instalar las dependencias almacenadas en `requirements.txt`
Para usarlo correctamente necesitas:
* Configurar tu correo personal
- Configurar la ruta absoluta de `SBATCH --chdir``
- Configurar las rutas absolutas de `BASE_DIR` y `VENV_DIR` .
> [!warning]
Las rutas que aparecen en la descripción del trabajo (`SBATCH --...)` DEBEN EXISTIR ANTES de que se ejecute el trabajo. De otro modo, el trabajo fallará sin devolver un mensaje explicativo.
Si tienes vinculado el correo recibirás un mensaje como `Slurm Job_id=1124194 Name=prepare_env Failed, Run time 00:00:00, FAILED, ExitCode 1`
 
Esta primera tarea se ejecutará en un nodo debug, que son nodos de corta duración y generalmente, tienen mucho menos trafico que los nodos standard. De forma que puedes detectar fallos de manera temprana sin esperar a que el trabajo encole.

```bash
#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=prepare_env
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=usuario@alumnos.upm.es

#SBATCH --output=logs/salida-%j.log
#SBATCH --error=logs/error-%j.log

# [[CONFIGURA]] esta ruta
#SBATCH --chdir=/home/x244/PROJECT/plantilla
##------------------------ End job description ------------------------

# 1. Definición de variables de ruta
BASE_DIR="/home/x244/PROJECT/plantilla"
VENV_DIR="${BASE_DIR}/entorno"


# 2. Carga de módulos
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

# 3. Gestión del entorno virtual
# Solo creamos el venv si no existe ya para ahorrar tiempo
if [ ! -d "$VENV_DIR" ]; then
    echo "Creando entorno virtual en $VENV_DIR..."
    python -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"

# 4. Instalación de dependencias
echo "Instalando dependencias..."
pip install -r "${BASE_DIR}/requirements.txt"

# Opcional: Verificar que las librerías están ahí
pip list

echo "Entorno preparado con éxito."

```

### 2. Ejecución del trabajo
Se encarga de:
- Activar el entorno virtual (asume que ya esta creado)
- Ejecutar el codigo `.py`
Para usarlo correctamente, al igual que con el anterior, debes configurar tu correo y las rutas absolutas.

Esta tarea se ejecuta en nodo standard gpu y usa 1 tarjeta V100.
```bash
#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=exec_job
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=usuario@alumnos.upm.es

#SBATCH --output=logs/salida-%j.log
#SBATCH --error=logs/error-%j.log

# Asegúrate de que esta ruta sea exacta
#SBATCH --chdir=/home/x244/PROJECT/plantilla
##------------------------ End job description ------------------------

# 1. Definición de variables de ruta
BASE_DIR="/home/x244/PROJECT/plantilla"
VENV_DIR="$(BASE_DIR)/entorno"

#2. Carga de modulos
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

#3. Activar el entorno
echo "Activando el entorno..."
source "${VENV_DIR}/bin/activate"

#Verificacion de seguridad
echo "GPUs detectadas por el sistema:"
nvidia-smi -L

#4. Lanzar el script
echo "Iniciando trabajo..."
python3 experimento.py

echo "Trabajo finalizado..."

```

