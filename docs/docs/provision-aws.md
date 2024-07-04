# Provision a good AWS EC2 instance
Select the NVIDIA deep learning AMI, on a g5.xlarge or g5.48xlarge instance. Do not forget to provision enough disk space too.

| Taille d'instance | GPU | Mémoire GPU (Gio) | vCPU | Mémoire (Gio) | Stockage d'instance (Go)         | Bande passante réseau (Gbit/s)*** | Bande passante EBS (Gbit/s) |
|-------------------|-----|-------------------|------|---------------|-----------------------------------|-----------------------------------|-----------------------------|
| g4dn.xlarge       | 1   | 4                 | 16   | 16            | 1 disque SSD NVMe de 125          | Jusqu'à 25                        | Jusqu'à 3,5                 |
| g5.xlarge         | 1   | 24                | 4    | 16            | 1 disque SSD NVMe de 250          | Jusqu'à 10                        | Jusqu'à 3,5                 |
| g5.48xlarge       | 8   | 192               | 192  | 768           | 2 SSD NVMe de 3 800               | 100                               | 19                          |

Allocate sufficient storage (e.g., 100GB).
