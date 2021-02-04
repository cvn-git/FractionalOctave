@flake8  FractionalOctave tests || goto :error

@dir /s /b FractionalOctave\*.py > filelist.txt
@dir /s /b tests\*.py >> filelist.txt
@mypy --ignore-missing-imports @filelist.txt || goto :error
@del filelist.txt

@REM pytest || goto :error

:error
@exit /b %errorlevel%
