
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

## Gestión de módulos (Lmod)
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

> [!info] Nota: Intenté cargar versiones más recientes de `apps/[año]` pero fallaron. Parece ser que la más reciente a fecha de hoy es `apps/2021`
