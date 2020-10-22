# SNOPT7.jl
Julia interface for SNOPT7

modified to have increased workspace size

# Linux install

copy "libsnopt7.so" to /usr/lib

copy "snopt7.lic" to /home/licenses

add to /.bashrc

export SNOPT_HOME="$HOME/usr/lib"

export SNOPT_LICENSE="$HOME/licenses/snopt7.lic"

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SNOPT_HOME}

export PATH="${SNOPT_HOME}:$PATH"
