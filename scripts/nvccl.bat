@echo off
REM 自动加载 MSVC 环境后调用 nvcc。
REM VS Code 的 build task 通过此脚本编译 .cu 文件，无需手动开 Developer Command Prompt。
call "D:\Visual Studio\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL
nvcc -Xcompiler "/utf-8" %*
