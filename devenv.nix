{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/basics/
  env.GREET = "Welcome to ann-suite development environment";

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.docker
    pkgs.stdenv.cc.cc.lib # for many python wheels on linux
    pkgs.bcc
    pkgs.linuxHeaders
    pkgs.glib
    pkgs.gcc # Explicitly adding GCC
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.12";
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  # Fix for python wheels needing libstdc++.so.6 and other common libs
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.glib
  ];

enterShell = ''
  echo $GREET
  python --version
  uv --version
  docker --version
  # Symlink for IDE compatibility
  if [ ! -L "$DEVENV_ROOT/.venv" ]; then
    ln -s "$DEVENV_STATE/venv" "$DEVENV_ROOT/.venv"
  fi
'';
}
