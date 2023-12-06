make install:
	kaggle datasets download -d rajeevw/ufcdata
	unzip ufcdata.zip
	rm ufcdata.zip
	rm data.csv
	rm raw_fighter_details.csv
	rm raw_total_fight_data.csv

make clean:
	rm preprocessed_data.csv