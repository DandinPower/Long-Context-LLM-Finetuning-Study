# install the latest 2.0.19 numactl for weighted-interleave

apt remove numactl
apt remove libnuma-dev

git clone https://github.com/numactl/numactl.git
./autogen.sh
./configure
make 
make test
make install

# set the weight

need to make sure the linux kernel is newer than kernel 6.9 to support the weighted_interleave version

<!-- rm -rf /sys/kernel/mm/mempolicy/weighted_interleave/
mkdir -p /sys/kernel/mm/mempolicy/weighted_interleave/
echo 1 > /sys/kernel/mm/mempolicy/weighted_interleave/node0 -->

