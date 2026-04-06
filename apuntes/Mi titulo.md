## A veces no deja instalar librerías
### Por ejemplo la librería accelerate

Tu entorno virtual en `/media/beegfs/...` es una instalación "ligera". No contiene el binario completo de Python, sino que apunta al Python del sistema. Al intentar ejecutar `pip`, el sistema busca `libpython3.11.so.1.0` para arrancar, pero como el administrador del clúster tiene Python en una ruta no estándar (dentro de `/software/...`), el sistema no la encuentra a menos que se la indiques explícitamente con `LD_LIBRARY_PATH`.

Solución: 
`module load Python/3.11.3-GCCcore-12.3.0`
`export LD_LIBRARY_PATH=$EBROOTPYTHON/lib:$LD_LIBRARY_PATH`
`source $entorno/bin/activate`
`pip install accelerate`
### encolar el trabajo
`sbatch job_scratch.sh`

### Verificar el estado del trabajo
`squeue -a`

PD: Pending. Estás en la cola esperando una GPU libre.
CF: Configuring. El nodo se está preparando para ti.
R: Running. Tu código se está ejecutando.
CG: Completing. El trabajo ha terminado y el sistema está limpiando los archivos temporales.

### Cancelar un trabajo
![[Pasted image 20260406122208.png]]
`scancel $JOBID`

### Para saber cuándo empieza tu trabajo
`squeue -j $JOBID --start`
Si aparece N/A es porque el cluster no sabe cuándo alguien puede liberar un gpu.
