IP=
D_DIR=
O_DIR=
O=
USER=
PWD=

get-dataset:
	wget -c -r -nd -nH -P $(D_DIR) --ftp-user=$(USER) --ftp-password=$(PWD) ftp://$(IP)/workspace/ftp/$(D_DIR)
	mv ./$(D_DIR)/* ./data/
	mkdir ./data/train/
	mkdir ./data/valid/
	mkdir ./data/test/
	mv ./data/train.txt ./data/train/
	mv ./data/valid.txt ./data/valid/
	mv ./data/test.txt ./data/test/
	rm -rf ./$(D_DIR)

train:
	python3 train.py

evalution:
	python3 evalution.py

upload:
	curl -T $(O) --ftp-create-dirs -u $(USER):$(PWD) ftp://$(IP)/workspace/ftp/$(O_DIR)/preprocess/$(O)

clean:
	rm -f ./checkpoint/*
	rm -f ./log/*