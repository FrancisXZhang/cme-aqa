@echo off
setlocal

cd /d "%~dp0\.."

if "%PYTHON%"=="" set PYTHON=python
if "%CME_AQA_DATA_ROOT%"=="" set "CME_AQA_DATA_ROOT=%CD%"

%PYTHON% tools\train_classification.py ^
  --fpv_json "%CME_AQA_DATA_ROOT%\output_p\A_FPV" --tpv_json "%CME_AQA_DATA_ROOT%\output_p\A_TPV" ^
  --fpv_f "%CME_AQA_DATA_ROOT%\output_v\A_FPV" --tpv_f "%CME_AQA_DATA_ROOT%\output_v\A_TPV" ^
  --label_file "%CME_AQA_DATA_ROOT%\Accu_extended.csv" ^
  --out_dir "%CD%\runs\class_accu_seed29082025" ^
  --num_epochs 20 ^
  --model_variant l2 ^
  --num_classes 9 ^
  --seed 29082025
if errorlevel 1 exit /b %errorlevel%

%PYTHON% tools\train_classification.py ^
  --fpv_json "%CME_AQA_DATA_ROOT%\output_p\A_FPV" --tpv_json "%CME_AQA_DATA_ROOT%\output_p\A_TPV" ^
  --fpv_f "%CME_AQA_DATA_ROOT%\output_v\A_FPV" --tpv_f "%CME_AQA_DATA_ROOT%\output_v\A_TPV" ^
  --label_file "%CME_AQA_DATA_ROOT%\Accu_extended.csv" ^
  --out_dir "%CD%\runs\class_accu_seed28082025" ^
  --num_epochs 20 ^
  --model_variant l2 ^
  --num_classes 9 ^
  --seed 28082025
if errorlevel 1 exit /b %errorlevel%

%PYTHON% tools\train_classification.py ^
  --fpv_json "%CME_AQA_DATA_ROOT%\output_p\A_FPV" --tpv_json "%CME_AQA_DATA_ROOT%\output_p\A_TPV" ^
  --fpv_f "%CME_AQA_DATA_ROOT%\output_v\A_FPV" --tpv_f "%CME_AQA_DATA_ROOT%\output_v\A_TPV" ^
  --label_file "%CME_AQA_DATA_ROOT%\Accu_extended.csv" ^
  --out_dir "%CD%\runs\class_accu_seed31082025" ^
  --num_epochs 20 ^
  --model_variant l2 ^
  --num_classes 9 ^
  --seed 31082025
if errorlevel 1 exit /b %errorlevel%

%PYTHON% tools\train_classification.py ^
  --fpv_json "%CME_AQA_DATA_ROOT%\output_p\A_FPV" --tpv_json "%CME_AQA_DATA_ROOT%\output_p\A_TPV" ^
  --fpv_f "%CME_AQA_DATA_ROOT%\output_v\A_FPV" --tpv_f "%CME_AQA_DATA_ROOT%\output_v\A_TPV" ^
  --label_file "%CME_AQA_DATA_ROOT%\Accu_extended.csv" ^
  --out_dir "%CD%\runs\class_accu_seed30082025" ^
  --num_epochs 20 ^
  --model_variant l2 ^
  --num_classes 9 ^
  --seed 30082025
if errorlevel 1 exit /b %errorlevel%

endlocal
exit /b 0
