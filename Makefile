IP=
D_DIR=
O_DIR=
O=
USER=
PWD=

get-dataset:
	wget -c -r -nd -nH -P $(D_DIR) --ftp-user=$(USER) --ftp-password=$(PWD) ftp://$(IP)/workspace/ftp/$(D_DIR)
	mkdir -p ./data/
	mkdir -p ./data/train/
	mkdir -p ./data/valid/
	mkdir -p ./data/test/
	mv ./$(D_DIR)/* ./data/
	mv ./data/train.txt ./data/train/
	mv ./data/valid.txt ./data/valid/
	mv ./data/test.txt ./data/test/
	rm -rf ./$(D_DIR)

train:
	python3 train.py

evalution:
	python3 evalution.py

upload:
	curl -T $(O) --ftp-create-dirs -u $(USER):$(PWD) ftp://$(IP)/workspace/ftp/$(O_DIR)

clean:
	rm -f ./expierments/base_model/checkpoint/*
	rm -f ./expierments/base_model/log/*