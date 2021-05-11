docker run -it --rm -p 8888:8888 -v $(dirname $(dirname `pwd`)):/$(basename $(dirname $(dirname `pwd`))) sample_embed_2d:pytorch-cpu
