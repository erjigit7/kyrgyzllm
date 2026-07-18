@echo off
rem Ночной прогон v3: дообучение числительных + дробей -> GGUF -> Ollama.
rem Запуск: двойной клик (или run_night_v3.cmd в терминале). Лог: night_v3.log.
rem Требование: свободная видеокарта (Photoshop и игры закрыты).
cd /d C:\Users\Admin20\Projects\KyrgyzLLM

echo [%time%] === v3: обучение === > night_v3.log
.venv\Scripts\python.exe -X utf8 legal_translate_train.py ^
  --adapter outputs/kazllm-legal-translate-v2-final ^
  --data legal_translate_aug.jsonl ^
  --output outputs/kazllm-legal-translate-v3 ^
  --lr 2e-5 --save-steps 300 >> night_v3.log 2>&1
if errorlevel 1 goto fail

echo [%time%] === v3: экспорт GGUF === >> night_v3.log
.venv\Scripts\python.exe -X utf8 legal_translate_export.py ^
  outputs/kazllm-legal-translate-v3-final outputs/kazllm-legal-translate-v3-gguf >> night_v3.log 2>&1
if errorlevel 1 goto fail

echo [%time%] === v3: обновление Ollama === >> night_v3.log
powershell -Command "(Get-Content Modelfile.translator) -replace 'v2-gguf_gguf', 'v3-gguf_gguf' | Set-Content Modelfile.v3" >> night_v3.log 2>&1
%LOCALAPPDATA%\Programs\Ollama\ollama.exe create myizam-translator -f Modelfile.v3 >> night_v3.log 2>&1
if errorlevel 1 goto fail

echo [%time%] === ГОТОВО: myizam-translator обновлён до v3 === >> night_v3.log
exit /b 0

:fail
echo [%time%] === ОШИБКА, смотри лог выше === >> night_v3.log
exit /b 1
