
## Acceder a Magerit
Se puede acceder usando el comando `ssh` con su usuario (en el ejemplo x123456) y el dominio `@magerit.cesvima.upm.es`. Luego se te pedirá la contraseña.

```bash 
usuario@PC:~/dir$ ssh x123456@magerit.cesvima.upm.es
(x123456@magerit.cesvima.upm.es) Password: ...
Last login: ...
[x123456@login2 ~]$
```

## Estructura de directorios

``` bash
/home/<code>/<user>/ : Directorio personal. Usado para almacenar configuraciones y datos personales.
/home/<code>/PROJECT/ : Directorio compartido entre miembros del proyecto. Usado para almacenar codigos, resultados o datos compartidos por varios miembros.
/home/<code>/SCRATCH/: Directorio temporal. Usado para almacenar logs y/o resultados parciales de ejecuciones.
```
Donde:
`<code>`:  Código del proyecto. Ej: x123
`<user>`: Código del usuario. Ej: x123456

# Gestión de módulos (Lmod)
Magerit gestiona las versiones de aplicaciones gracias a Lmod. A continuación, comandos útiles para manejar estos módulos.
### Búsqueda de módulos: module keyword
`module keyword <key1> <key2> <key3>` permite buscar los módulos disponibles usando una o varias palabras claves. Es más flexible buscando que `module avail`. 
Especialmente útil cuando no se conoce el nombre del módulo.
```bash
[x244645@login2 ~]$ module keyword python Python python3
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The following modules match your search criteria: "python", "Python", "python3"
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  ANTLR: ANTLR/2.7.7-GCCcore-11.3.0-Java-11, ANTLR/2.7.7-GCCcore-12.3.0-Java-11
    ANTLR, ANother Tool for Language Recognition, (formerly PCCTS) is a language tool that provides a framework for constructing recognizers, compilers, and translators from
    grammatical descriptions containing Java, C#, C++, or Python actions. 

  AlphaFold3: AlphaFold3/3.0.1-foss-2023a-CUDA-12.1.1
    Bundle of Python packages for use with AlphaFold3

  Anaconda3: Anaconda3/2024.02-1, Anaconda3/2025.06-1
    Built to complement the rich, open source Python community, the Anaconda platform provides an enterprise-ready data analytics platform that empowers companies to adopt a modern
    open data science analytics architecture. 

  BeautifulSoup: BeautifulSoup/4.10.0-GCCcore-11.3.0, BeautifulSoup/4.12.2-GCCcore-12.3.0, BeautifulSoup/4.12.2-GCCcore-13.2.0
    Beautiful Soup is a Python library designed for quick turnaround projects like screen-scraping.

...

```

### Ver módulos disponibles: module avail 
`module avail <app>` muestra todas las versiones disponibles asociadas a una aplicación del sistema.
> [!info] Nota: Este comando puede tardar un momento en cargar...

```bash
[x123456@login2 ~]$ module avail Python/3

-------------------------------------------------------------------------- /media/apps/avx512-2021/modules/all --------------------------------------------------------------------------
   Python/3.6.6-foss-2018b        Python/3.9.5-GCCcore-10.3.0-bare     Python/3.10.8-GCCcore-12.2.0-bare    Python/3.13.5-GCCcore-14.3.0                   (D)
   Python/3.6.6-fosscuda-2018b    Python/3.9.5-GCCcore-10.3.0          Python/3.10.8-GCCcore-12.2.0         protobuf-python/3.13.0-foss-2020a-Python-3.8.2
   Python/3.7.2-GCCcore-8.2.0     Python/3.9.6-GCCcore-11.2.0-bare     Python/3.11.3-GCCcore-12.3.0         protobuf-python/3.14.0-GCCcore-10.2.0
   Python/3.7.4-GCCcore-8.3.0     Python/3.9.6-GCCcore-11.2.0          Python/3.11.5-GCCcore-13.2.0         protobuf-python/3.17.3-GCCcore-10.3.0
   Python/3.8.2-GCCcore-9.3.0     Python/3.10.4-GCCcore-11.3.0-bare    Python/3.12.3-GCCcore-13.3.0         protobuf-python/3.17.3-GCCcore-11.2.0
   Python/3.8.6-GCCcore-10.2.0    Python/3.10.4-GCCcore-11.3.0         Python/3.13.1-GCCcore-14.2.0         protobuf-python/3.19.4-GCCcore-11.3.0

  Where:
   D:  Default Module

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".

```

### Ver módulos cargados: module list
`module list` te proporciona una lista de los nodos cargados actualmente.
Por defecto, se cargan: `StdEnv` y `apps/2021`.

```bash
[x244645@login3 ~]$ module list

Currently Loaded Modules:
  1) StdEnv               5) ncurses/6.5-GCCcore-14.3.0     9) libiconv/1.18-GCCcore-14.3.0     13) cURL/8.14.1-GCCcore-14.3.0     17) lz4/1.10.0-GCCcore-14.3.0
  2) apps/2021      (S)   6) zlib/1.3.1-GCCcore-14.3.0     10) libunistring/1.3-GCCcore-14.3.0  14) XZ/5.8.1-GCCcore-14.3.0        18) zstd/1.5.7-GCCcore-14.3.0
  3) CUDA/11.3.1          7) bzip2/1.0.8-GCCcore-14.3.0    11) libpsl/0.21.5-GCCcore-14.3.0     15) libxml2/2.14.3-GCCcore-14.3.0  19) libarchive/3.8.1-GCCcore-14.3.0
  4) GCCcore/14.3.0       8) libidn2/2.3.8-GCCcore-14.3.0  12) OpenSSL/3                        16) gzip/1.14-GCCcore-14.3.0       20) CMake/4.0.3-GCCcore-14.3.0

  Where:
   S:  Module is Sticky, requires --force to unload or purge

 

```

### Carga y descarga: module load y module unload
`module load <app>` carga la versión por defecto de la aplicación; esta versión esta marcada con una (D). Para cargar una versión en concreto, se usa `module load <app/version>`
`module unload <app>` permite descargar la aplicación.
Para visualizar la carga y descarga, podemos usar `module list` entre ejecuciones.

Estado inicial:
``` bash
[x123456@login3 ~]$ module list

Currently Loaded Modules:
  1) StdEnv   2) apps/2021 (S)

  Where:
   S:  Module is Sticky, requires --force to unload or purge
```
Carga: CUDA
``` bash
[x123456@login3 ~]$ module load CUDA
```
Estado actual:
```bash
[x123456@login3 ~]$ module list

Currently Loaded Modules:
  1) StdEnv   2) apps/2021 (S)   3) CUDA/11.3.1

  Where:
   S:  Module is Sticky, requires --force to unload or purge

```
Descarga de CUDA:
``` bash
[x123456@login3 ~]$ module unload CUDA/11.3.1 
```
Estado final:
``` bash
[x244645@login3 ~]$ module list

Currently Loaded Modules:
  1) StdEnv   2) apps/2021 (S)

  Where:
   S:  Module is Sticky, requires --force to unload or purge
```

Magerit tiene disponible una serie de compilaciones preparadas por año, las cuales carga por defecto. En nuestras máquinas, a fecha de hoy, se tiene cargado `apps/2021`. Se puede seleccionar una de ellas con `module load apps/[año]`

> [!info] NOTA
> Intenté cargar versiones más recientes de `apps/[año]` pero fallaron. Parece ser que la más reciente a fecha de hoy es `apps/2021`

### Limpieza de módulos: module purge
Podemos limpiar todos los módulos presentes en un nodo con el comando `module purge`. La única excepción es `apps/2021` que solo se puede eliminar usando la opción `--force`

Antes de purge:
```bash 
[x244645@login1 ~]$ module list

Currently Loaded Modules:
  1) StdEnv   2) apps/2021 (S)

  Where:
   S:  Module is Sticky, requires --force to unload or purge
```
Purge:
```bash
[x244645@login1 ~]$ module purge
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) apps/2021
```
Después de purge:
```bash
[x244645@login1 ~]$ module list

Currently Loaded Modules:
  1) apps/2021 (S)

  Where:
   S:  Module is Sticky, requires --force to unload or purge
```

Para limpiar completamente los módulos, usamos `module --force purge`
```bash
[x244645@login1 ~]$ module --force purge
[x244645@login1 ~]$ module list
No modules loaded
```

```bash
#Este comando no funciona
[x244645@login1 ~]$ module purge --force
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) apps/2021

```

# Sobreviviendo al "Version hell"
## Versiones de Python
Magerit ofrece múltiples versiones de Python, lo que puede resultar confuso al inicio. Por lo que a continuación, explicaré varios puntos sobre las versiones disponibles.

```bash
[x123456@login1 x123]$ module avail Python/3

----------------------------------------------------------------------------- /media/apps/avx512-2021/modules/all -----------------------------------------------------------------------------
   Python/3.6.6-foss-2018b        Python/3.9.5-GCCcore-10.3.0-bare     Python/3.10.8-GCCcore-12.2.0-bare  Python/3.13.5-GCCcore-14.3.0 (D)
   Python/3.6.6-fosscuda-2018b    Python/3.9.5-GCCcore-10.3.0          Python/3.10.8-GCCcore-12.2.0      protobuf-python/3.13.0-foss-2020a-Python-3.8.2
   Python/3.7.2-GCCcore-8.2.0     Python/3.9.6-GCCcore-11.2.0-bare     Python/3.11.3-GCCcore-12.3.0         protobuf-python/3.14.0-GCCcore-10.2.0
   Python/3.7.4-GCCcore-8.3.0     Python/3.9.6-GCCcore-11.2.0          Python/3.11.5-GCCcore-13.2.0         protobuf-python/3.17.3-GCCcore-10.3.0
   Python/3.8.2-GCCcore-9.3.0     Python/3.10.4-GCCcore-11.3.0-bare    Python/3.12.3-GCCcore-13.3.0         protobuf-python/3.17.3-GCCcore-11.2.0
   Python/3.8.6-GCCcore-10.2.0    Python/3.10.4-GCCcore-11.3.0         Python/3.13.1-GCCcore-14.2.0         protobuf-python/3.19.4-GCCcore-11.3.0

  Where:
   D:  Default Module
```

### Toolchain
Después de la versión de Python `Python/3.XX.X` aparece `GCCcore, foss, o fosscuda`
- `GCCcore` es la más común. Se construyo una versión especifica del compilador GCC. Se recomienda si usas librerías que se compilen en C++, como `numpy o pandas`
- `foss (Free Open Source Software)` incluye el compilador GCC y además librerías matemáticas optimizadas. Se recomienda para calculo numérico pesado.
- `fosscud` es similar a foss, pero esta pre configurado para hacer uso de `CUDA`. Recomendable si se usa `PyTorch` o `TensorFlow` con GPU.
### Sufijo -bare
Algunas versiones incluyen el sufijo `-bare`, como por ejemplo: `Python/3.9.6-GCCcore-11.2.0-bare` . Estos significa que es una instalación desnuda. Solo tiene instalado lo mínimo necesario para funcionar.
Se recomienda usar este tipo de módulos para controlar al máximo que versión de cada librería se usa, aunque suelen traer consigo problemas si no se organiza correctamente la instalación de dependencias.
### Versión por defecto
Por defecto, a fecha de hoy (marzo de 2026), se ofrece por defecto la versión `Python/3.13.5-GCCcore-14.3.0`. La versión por defecto esta marcada por `(D)`

### Como elegir la versión: Puntos a tener en cuenta
- Si desarrollas código en tu ordenador personal, puedes obtener la versión que tienes escribiendo:
```bash
usuario@PC:~$ python --version
Python 3.12.3
```
Esto te puede dar una idea sobre que versión usar. Por ejemplo, si en mi ordenador trabajo con la versión `3.12.3` , la elección más cercana a mi entorno sería `Python/3.12.3-GCCcore-13.3.0`

- Usar la ultima versión puede ser contraproducente, debido a que muchas librerías tardan en adaptarse a los cambios de sintaxis y a la estructura de ejecución del programa.  Al ser un lenguaje interpretado, Python puede sufrir cambios significativos de una versión a otra.
## Entornos virtuales (venvs)
Los entornos virtuales o venvs son una de las herramientas más poderosas que tiene Python para manejar las versiones a lo largo de diferentes proyectos y dentro de Magerit, usarlas es un requisito. Dado que es imposible usar una "configuración global" de Python que todos los usuarios puedan usar sin tener conflictos de dependencias, cada proyecto y/o usuario debe gestionar sus propias dependencias dentro de un venv.
Podemos pensar en un venv como una "burbuja": un entorno aislado en el que convive una versión especifica de Python y sus librerías, junto a utilidades (como pip).  A modo practico, un venv es un directorio en el que residen los binarios de Python y sus librerias.

### Creación del entorno
Puedes crear un venv usando el siguiente comando. **Esto solo se realiza una vez.**
```bash 
[x123456@login3 prueba_venv]$ python3 -m venv entorno_prueba
```
Donde `entorno_prueba` es el nombre del entorno, es personalizable.

> [!warning] IMPORTANTE
> Antes de crear un venv es importante cargar el modulo de Python que vayamos a utilizar, usando `module load Python/3....`.
>  Si no cargamos ningún modulo, el comando utilizará la versión por defecto de Python que tenga definida (no tiene porque ser la misma versión por defecto que se lista en `module avail Python/3`) y podemos tener conflictos al instalar librerías.

Una vez activado el entorno (como se ve en el siguiente punto), podemos comprobar que versión de Python esta usando el venv.
```bash
(entorno_prueba) [x123456@login3 prueba_venv]$ python --version
Python 3.6.8
```

### Activación del entorno
Una vez creado el entorno, podemos activarlo usando el siguiente comando. El comando le indica al sistema que deseamos usar la configuración de Python que se encuentra en este entorno. 
**Esto se debe realizar cada vez que queramos usar el entorno**
```bash 
[x123456@login3 prueba_venv]$ source entorno_prueba/bin/activate
(entorno_prueba) [x123456@login3 prueba_venv]$ 
```

Como podemos ver, una vez ejecutado el comando de activación aparece entre paréntesis el nombre del entorno: `(entorno_prueba)`
### Trabajo en el entorno
Una vez activado el entorno, podemos instalar las dependencias que queramos. En el siguiente apartado, explicaremos como se trabaja con `pip`.
### Desactivación del entorno
Cuando hemos terminado de trabajar con el entorno, podemos "salir" de la burbuja o desactivar el entorno con el comando:
```bash
(entorno_prueba) [x123456@login3 prueba_venv]$ deactivate
[x123456@login3 prueba_venv]$ 
```

> [!info] NOTA
> Al enviar trabajos a Magerit, realmente no es necesario desactivar el venv porque toda la sesión se destruye al finalizar el trabajo.
> Aún así se considera una buena práctica de limpieza, por lo que es recomendable usarlo en todos nuestros proyectos.

### Eliminar el entorno
Aunque es muy poco probable que tengas que hacer esto, puedes eliminar un entorno virtual simplemente borrando el directorio que lo contiene.
``` bash
[x123456@login3 prueba_venv]$ rm -rf entorno_prueba/
```

## Utilidades de `pip`

Pip es el gestor de librerias de Python por excelencia. Curiosamente, es el acrónimo de `Pip Installs Packages`, por lo que es un acrónimo recursivo.

Su funcionamiento es el siguiente, si yo ejecuto `pip install pandas`:
1. Busca "pandas" en el servidor de PyPi (Python Package Index)
2. Comprueba las dependencias de "pandas"
3. Descarga e instala todo lo necesario dentro del entorno virtual

> [!warning] AVISO
> Solo se puede usar una versión de cada librería a la vez, por lo que es extremadamente importante saber que versiones usamos. De ese modo, evitamos conflictos de dependencias
### Instalar última versión de una librería
Se puede instalar la ultima versión de una librería indicando únicamente su nombre. Por ejemplo, `pip install pandas`

```bash
(entorno_prueba) [x244645@login2 prueba_venv]$ pip install pandas
Collecting pandas
  Downloading https://files.pythonhosted.org/packages/c3/e2/00cacecafbab071c787019f00ad84ca3185952f6bb9bca9550ed83870d4d/pandas-1.1.5-cp36-cp36m-manylinux1_x86_64.whl (9.5MB)
    100% |████████████████████████████████| 9.5MB 137kB/s 
Collecting numpy>=1.15.4 (from pandas)
  Cache entry deserialization failed, entry ignored
  Downloading https://files.pythonhosted.org/packages/45/b2/6c7545bb7a38754d63048c7696804a0d947328125d81bf12beaa692c3ae3/numpy-1.19.5-cp36-cp36m-manylinux1_x86_64.whl (13.4MB)
    100% |████████████████████████████████| 13.4MB 94kB/s 
Collecting pytz>=2017.2 (from pandas)
  Downloading https://files.pythonhosted.org/packages/10/99/781fe0c827be2742bcc775efefccb3b048a3a9c6ce9aec0cbf4a101677e5/pytz-2026.1.post1-py2.py3-none-any.whl (510kB)
    100% |████████████████████████████████| 512kB 1.3MB/s 
Collecting python-dateutil>=2.7.3 (from pandas)
  Downloading https://files.pythonhosted.org/packages/ec/57/56b9bcc3c9c6a792fcbaf139543cee77261f3651ca9da0c93f5c1221264b/python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229kB)
    100% |████████████████████████████████| 235kB 1.1MB/s 
Collecting six>=1.5 (from python-dateutil>=2.7.3->pandas)
  Downloading https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl
Installing collected packages: numpy, pytz, six, python-dateutil, pandas
 
Successfully installed numpy-1.19.5 pandas-1.1.5 python-dateutil-2.9.0.post0 pytz-2026.1.post1 six-1.17.0
You are using pip version 9.0.3, however version 26.0.1 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
```

Como podemos ver, al instalar `pandas`, pid trae consigo ciertas dependencias como `numpy, pytz, python-dateutil y siz`

> [!info] NOTA
>  Instalar dependencias desde la terminal interactiva de Magerit es extremadamente lento, para realizar la preparación del entorno virtual se recomienda usar los "nodos debug"
### Instalar versión especifica de una librería
Para instalar una versión especifica debemos indicarla después del nombre. Por ejemplo, `pip install numpy==1.15.4`

### Ver las librerias instaladas
Podemos verlas usando el comando `pip list`. Podemos indicar el formato deseado.
```bash
(entorno_prueba) [x123456@login2 prueba_venv]$ pip list
DEPRECATION: The default format will switch to columns in the future. You can use --format=(legacy|columns) (or define a format=(legacy|columns) in your pip.conf under the [list] section) to disable this warning.
numpy (1.19.5)
pandas (1.1.5)
pip (9.0.3)
python-dateutil (2.9.0.post0)
pytz (2026.1.post1)
setuptools (39.2.0)
six (1.17.0)

(entorno_prueba) [x123456@login2 prueba_venv]$ pip list --format=columns
Package         Version     
--------------- ------------
numpy           1.19.5      
pandas          1.1.5       
pip             9.0.3       
python-dateutil 2.9.0.post0 
pytz            2026.1.post1
setuptools      39.2.0      
six             1.17.0      
```

### Eliminar una librería
Si cometimos un error o ya no queremos usar una librería, podemos eliminarla indicando su nombre. Por ejemplo, `pip uninstall pandas`

```bash
(entorno_prueba) [x123456@login2 prueba_venv]$ pip uninstall pandas
```

### "Congelar" las librerías instaladas
Una de los comandos más útiles para trabajar con este tipo de entornos es `pip freeze`. Este comando vuelca la lista de dependencias instaladas en el entorno en un momento dado.
```bash
(entorno_prueba) [x123456@login2 prueba_venv]$ pip freeze
numpy==1.19.5
pandas==1.1.5
python-dateutil==2.9.0.post0
pytz==2026.1.post1
six==1.17.0
```

Por convención, **este volcado se suele redirigir a un fichero llamado `requirements.txt`**
```bash
(entorno_prueba) [x123456@login2 prueba_venv]$ pip freeze > requirements.txt
```

### Replicar un entorno a partir de `requirements.txt`
Una vez construido el fichero `requirements.txt`, podemos exportar o replicar un entorno descargando todas sus dependencias. Se usa el comando `pip install -r requirements.txt`

Para este ejemplo, crearé un nuevo entorno y replicaré las dependencias de `entorno_prueba`
```bash
[x123456@login2 prueba_venv]$ python3 -m venv entorno_copiado
[x123456@login2 prueba_venv]$ source entorno_copiado/bin/activate
(entorno_copiado) [x123456@login2 prueba_venv]$ pip install -r requirements.txt 
Collecting numpy==1.19.5 (from -r requirements.txt (line 1))
  Using cached https://files.pythonhosted.org/packages/45/b2/6c7545bb7a38754d63048c7696804a0d947328125d81bf12beaa692c3ae3/numpy-1.19.5-cp36-cp36m-manylinux1_x86_64.whl
Collecting pandas==1.1.5 (from -r requirements.txt (line 2))
  Using cached https://files.pythonhosted.org/packages/c3/e2/00cacecafbab071c787019f00ad84ca3185952f6bb9bca9550ed83870d4d/pandas-1.1.5-cp36-cp36m-manylinux1_x86_64.whl
Collecting python-dateutil==2.9.0.post0 (from -r requirements.txt (line 3))
  Using cached https://files.pythonhosted.org/packages/ec/57/56b9bcc3c9c6a792fcbaf139543cee77261f3651ca9da0c93f5c1221264b/python_dateutil-2.9.0.post0-py2.py3-none-any.whl
Collecting pytz==2026.1.post1 (from -r requirements.txt (line 4))
  Using cached https://files.pythonhosted.org/packages/10/99/781fe0c827be2742bcc775efefccb3b048a3a9c6ce9aec0cbf4a101677e5/pytz-2026.1.post1-py2.py3-none-any.whl
Collecting six==1.17.0 (from -r requirements.txt (line 5))
  Using cached https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl
Installing collected packages: numpy, pytz, six, python-dateutil, pandas
Successfully installed numpy-1.19.5 pandas-1.1.5 python-dateutil-2.9.0.post0 pytz-2026.1.post1 six-1.17.0
You are using pip version 9.0.3, however version 26.0.1 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
(entorno_copiado) [x123456@login2 prueba_venv]$ 

```


# Definición de trabajos con `sbatch`
"Magerit se explota mediante trabajos `batch` usando SLURM como gestor y planificador de recursos". Esto significa que nosotros debemos encargarnos de la configuración del trabajo y dependiendo del trabajo, el planificador asignará uno o varios nodos para realizarlo.
Los trabajos se definen a través de un fichero `job.sh` (un script de shell), que contiene las directivas de SLURM junto con los pasos de preparación del entorno y la llamada de ejecución de nuestro programa.

Se nos proporciona la siguiente plantilla de `job.sh`
```bash
#!/bin/bash -l
##----------------------- Start job description -----------------------
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=nombre_trabajo
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=usuario@ejemplo.com
#SBATCH --output=ruta_absoluta/run_logs/out-%u-%x-%j.log
#SBATCH --error=ruta_absoluta/run_logs/err-%u-%x-%j.log
#SBATCH --chdir=ruta_absoluta

##------------------------ End job description ------------------------

module purge && module load <app>

srun <app> --app-param app_args
```

> [!info] NOTA
> Se recomienda revisa la guía oficial de Cesvima
> https://docs.cesvima.upm.es/magerit/jobs/

## Encolar el trabajo
Podemos encolar un trabajo usando el comando  `sbatch job_scratch.sh`
## Verificar el estado del trabajo
Podemos verificar el estado de los trabajos que hayamos lanzado usando el mandato `squeue -a`
En la columna ST (state) pueden aparecen las siguientes siglas:
```bash
PD: Pending. Estás en la cola esperando una GPU libre.
CF: Configuring. El nodo se está preparando para ti.
R: Running. Tu código se está ejecutando.
CG: Completing. El trabajo ha terminado y el sistema está limpiando los archivos temporales.
```
### Cancelar un trabajo
Puedes cancelar un trabajo encolado / en ejecución con el comando `scancel $JOBID`, donde `JOBID` es el id del trabajo.

![[Pasted image 20260406122208.png]]
## Para estimar cuándo empezará tu trabajo
`squeue -j $JOBID --start`
Si aparece `N/A` es porque el cluster no sabe cuándo alguien puede liberar un gpu.

# Miscelánea
## Problemas al instalar librerias
 A veces no deja instalar librerías, por ejemplo la librería accelerate
 
> Tu entorno virtual en `/media/beegfs/...` es una instalación "ligera". No contiene el binario completo de Python, sino que apunta al Python del sistema. Al intentar ejecutar `pip`, el sistema busca `libpython3.11.so.1.0` para arrancar, pero como el administrador del clúster tiene Python en una ruta no estándar (dentro de `/software/...`), el sistema no la encuentra a menos que se la indiques explícitamente con `LD_LIBRARY_PATH`.
#### Solución
Indicar manualmente la ruta que debe usar:
```bash
module load Python/3.11.3-GCCcore-12.3.0
export LD_LIBRARY_PATH=$EBROOTPYTHON/lib:$LD_LIBRARY_PATH
source $entorno/bin/activate
pip install accelerate
```
