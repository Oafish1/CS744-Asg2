# Initial Setup
### Local Modules
Install parallel-ssh
```bash
pip install parallel-ssh
```

Create an additional file `nodes` that stores the host addresses of each node.

### Node Installation
Set up `parallel-ssh`
```bash
alias pssh="parallel-ssh -h nodes -i -x '-i ~/.ssh/id_cloudlab $1'"
```

Install updates
```bash
pssh sudo apt update
pssh wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Install Miniconda individually and add to path
```bash
sh Miniconda3-latest-Linux-x86_64.sh
```

Install other modules
```bash
pssh <USER DIR>/miniconda3/bin/conda install numpy && \
pssh <USER DIR>/miniconda3/bin/conda install pytorch torchvision torchaudio cpuonly -c pytorch
# Enter 'y' twice on command line
```

Create dir
```bash
pssh mkdir programs
```

# Starting the Nodes
### Website
`https://www.cloudlab.us/status.php?uuid=<UUID>`

### Connecting
Connect to main node
```bash
ssh <USER NAME>@<CLUSTER ID>.wisc.cloudlab.us -i ~/.ssh/id_cloudlab -p 27010
```


# Running Scripts
### Setup
Copy over necessary files
```bash
scp -P 27010 -i ~/.ssh/id_cloudlab * <USER NAME>@<CLUSTER ID>.wisc.cloudlab.us:~/programs && \
scp -P 27011 -i ~/.ssh/id_cloudlab * <USER NAME>@<CLUSTER ID>.wisc.cloudlab.us:~/programs && \
scp -P 27012 -i ~/.ssh/id_cloudlab * <USER NAME>@<CLUSTER ID>.wisc.cloudlab.us:~/programs && \
scp -P 27013 -i ~/.ssh/id_cloudlab * <USER NAME>@<CLUSTER ID>.wisc.cloudlab.us:~/programs
```

### Part 1
Run on one node
```bash
python main.py
```

### All Others
Run on each node individually
```bash
python programs/main.py --master-ip 10.10.1.1 --num-nodes 4 --rank 0
python programs/main.py --master-ip 10.10.1.1 --num-nodes 4 --rank 1
python programs/main.py --master-ip 10.10.1.1 --num-nodes 4 --rank 2
python programs/main.py --master-ip 10.10.1.1 --num-nodes 4 --rank 3
```
