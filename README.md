Amazon ML Hacks :

# Commands To Create Environment
- pip install virtualenv env
- virtualenv env 
- env/scripts/activate
- pip install -r requirements.txt

# Guide to Test The File
- Update the parameters.json file with the input path and output path.
- Change the "testpath" field with the path where the test/input file is located. 
- Change the "outputpath" field with the path where you want the output to stored.
- Run the main.py script to run the test (csv file) thorugh the trained model and get the corresponding results. 
<pre><code>Command: !python main.py</code></pre> 
