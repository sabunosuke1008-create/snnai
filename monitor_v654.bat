@echo off
:loop
python -m kaggle kernels status gihuhi/snnai-v6-5-4 > monitor_v654_status.txt 2>&1
findstr /C:"KernelWorkerStatus.COMPLETE" /C:"KernelWorkerStatus.ERROR" monitor_v654_status.txt > nul
if %errorlevel%==0 goto done
timeout /t 300 /nobreak > nul
goto loop
:done
python -m kaggle kernels logs gihuhi/snnai-v6-5-4 > kaggle_output_v654/kaggle_kernel.log 2>&1
echo DONE > monitor_v654_done.txt
