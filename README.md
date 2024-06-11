## Stytra Analysis 

The following package is for analyzing data collected from Stytra. The 
Preprocessing script can be used for both freely-swimming and head-fixed
fish experiments. The Analysis script is meant for freely-swimming.

### Installation 

This repository requires Python 3.12 or higher. 

Clone this repository. Then, you'll need to create a virtual environment 
to be able to run the programs: 
```
cd stytra-analysis
python3 -m venv venv 
source venv/bin/activate 
pip install -r requirements.txt
```
You should now have all the dependencies. Anytime you run this software, 
start by activating the virtual environment by doing this:
```
cd stytra-analysis
source venv/bin/activate
```

### Preprocessing

Before generating analyses, we need to preprocess your raw experimental data:
```
python preprocess.py /path/to/experiment --trials TRIAL
```
The `--trials` parameter is optional. It specifies the number of experiments 
per repeats that the analysis should consider aggregations over. If it is not 
specified, it will be automatically set to 3. The path to the experiment should be an 
absolute path, not a relative path. 
If you are on a MacBook, you can just drag and drop the folder into terminal.

To see more information, you can run:
```
python preprocess.py --help
```

### Analysis 
Now that preprocessing is done, we can run `analysis.py` to generate plots. 
It can be run much like preprocessing:
```
python analysis.py /path/to/experiment
```
Running the above command will create all of the plots you need in a folder in 
that data directory called `plots`. It will also generate a `latencies.csv` file. 
Remember, this must be run with the virtual environment enabled. Additionally, 
the path to the experiment should be an absolute path, not a relative path. 
