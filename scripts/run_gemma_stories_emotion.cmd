@echo off
setlocal
cd /d "%~dp0.."
if not exist "test data\output\gemma_emotion_stories" mkdir "test data\output\gemma_emotion_stories"
echo started %DATE% %TIME% > "test data\output\gemma_emotion_stories\run_status.txt"
call ".\.venv\Scripts\python.exe" "src\test_data\evaluate_gemma_emotion_stories.py" > "test data\output\gemma_emotion_stories\run_stdout.log" 2> "test data\output\gemma_emotion_stories\run_stderr.log"
echo exit_code %ERRORLEVEL% > "test data\output\gemma_emotion_stories\exit_code.txt"
echo finished %DATE% %TIME% >> "test data\output\gemma_emotion_stories\run_status.txt"
endlocal
