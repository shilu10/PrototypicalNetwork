!wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip
!wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip

if [ -d "data" ];
then 
	echo "directory exists"
else
	mkdir "data"

fi

cd data

unzip images_background.zip
unzip images_evaluation.zip


