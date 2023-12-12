import os

def test_analysis():
	os.chdir("../")
	os.system("python3 run_analysis.py -d bray-curtis -i ./tests -o ./tests -f mock_genus_data.csv") # run with bray-curtis distance
	os.system("python3 run_analysis.py -d jaccard -i ./tests -o ./tests -f mock_genus_data.csv") # run with jaccard distance

test_analysis()