@echo off
setlocal

cd /d "%~dp0\.."

if "%PYTHON%"=="" set PYTHON=python
if "%CME_AQA_DATA_ROOT%"=="" set "CME_AQA_DATA_ROOT=%CD%"

%PYTHON% tools\train_classification.py ^
  --fpv_json "%CME_AQA_DATA_ROOT%\output_p\T_FPV" --tpv_json "%CME_AQA_DATA_ROOT%\output_p\T_TPV" ^
  --fpv_f "%CME_AQA_DATA_ROOT%\output_v\T_FPV" --tpv_f "%CME_AQA_DATA_ROOT%\output_v\T_TPV" ^
  --label_file "%CME_AQA_DATA_ROOT%\Tuina_extended.csv" ^
  --out_dir "%CD%\runs\class_tuina_seed42" ^
  --num_epochs 20 ^
  --model_variant l2 ^
  --num_classes 9 ^
  --seed 42
if errorlevel 1 exit /b %errorlevel%

endlocal
exit /b 0
