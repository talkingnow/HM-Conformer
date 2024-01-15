sudo docker run --gpus all -it --rm --ipc=host -v {PATH_DB}:/data -v \
{PATH_HM-Conformer}/env202305:/environment -v \
{PATH_HM-Conformer}/env202305/results:/results -v \
{PATH_HM-Conformer}/exp_lib:/exp_lib -v \
{PATH_HM-Conformer}:/code env202305:latest