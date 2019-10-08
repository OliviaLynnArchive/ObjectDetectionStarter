# Object Detection Starter
## Installation Instructions
This was a little messy.

A big part of the problem is the fact that the Tensorflow Object Detection API is not yet compatible with the current version of TensorFlow, version 2.0. This kind of incompatibility problem trickles down to all the dependencies, and many published articles, tutorials, and walkthroughs (or perhaps, only the ones that I unfortunately stumbled upon) list a couple certain versions of tools they use, but default to the most recent versions of other dependencies. 

I hit more than one dead end that felt a lot like the fox, chicken, and corn river-crossing problem. The author says protobuf versions above 3.6.0 are unstable? No problem. You’ve installed python 3.6 so you could install tensorflow-gpu 1.9, only to discover tf 1.9 requires protobuf 3.6.1? Great.

The main tutorial I used was Lyudmil Vladimirov’s [TensorFlow Object Detection API Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)--but naturally, with some key changes to get things working on my machine.

(For the record, I'm on Windows 10)

### CUDA, CuDNN, and Anaconda
These, I installed as per the directions.

* CUDA Toolkit v9.0
* CuDNN v7.0.5
* Anaconda with Python 3.7

Note: I think for the version I ended up doing, I did not end up updating my driver after installing CUDA for the final time. I haven’t run into any problems with that, yet.

### Protobuf
For this one, I ended up diverging from the tutorial.

* Protobuf 3.9.2

I initially followed the tutorial’s directions around protobuf, but by the time I got things working I saw that I had protobuf v 3.9.2 when calling conda list.

I guess I ended up reinstalling a more recent version (I know at some point I was using protobuf v 3.5.1 as the article said, but I think it was the case that an article somewhere else mentioned a more recent one working for them)

### Python
The major differences to note are Python 3.5 and Tensorflow-gpu 1.5!

* python 3.5
* tensorflow-gpu 1.5
* (I didn’t bother with lxml or jupyter)
* opencv (didn’t specify, defaulted to 3.4.3)
* matplotlib (defaulted to 2.2.2 - didn’t end up using this bc of a pyqt warning, TODO)
* pillow (defaulted to 5.3.0)

For convenience, the commands I used were:

```
conda create -n tf15 python=3.5
activate tf15
pip install --ignore-installed --upgrade tensorflow-gpu=1.5
conda install opencv 
conda install matplotlib 
conda install pillow
```

It’s worth mentioning that I was able to include matplotlib without a warning about pyqt in a previous conda env. I believe this is probably because I had tried to follow the section for the imgLabel tool previously, and while I couldn’t get it to actually work, this may have set my pyqt to something matplotlib was looking for.

### Extras
I have not yet explored the COCO API installation, and when I followed the instructions for the labelImg Installation, I kept getting conda error messages about different packages being incompatible with pyqt v4, so it wouldn’t let me make a new virtual environment with pyqt=4 that labelImg needs. I might try out the precompiled binary instead.

### The Code
I used the code provided by Vladimirov in his tutorial, but I kept running into an error with the line:
`PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')`
where it apparently wasn’t a valid path name, or something along that line.

I tried out an absolute path, even uninstalled Anaconda from “C:\Users\Olivia Lynn\Anaconda” and reinstalled it to “C:\ProgramData\Anaconda” because Anaconda is notoriously unstable for paths that involve spaces, but ultimately, the fix was to add an ‘r’ before a string containing the absolute path, making it a raw string. So, change that line to:
`PATH_TO_LABELS = r'C:\TensorFlow\models\research\object_detection\data\mscoco_label_map.pbtxt'`
or equivalent for your directory structure.

### Other References Worth Mentioning
* [LabelImg](https://github.com/tzutalin/labelImg) The next step from here.
* [EdjeElectronics Article](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10): A very thourough walkthrough, but I only followed it to partway through. IIRC, I made it through the initial tf test, but wanted to see a working object detector on my computer before taking an afternoon to take and label my own images. I like that this has a detailed explanation of creating your own dataset and training your own model, and I plan to revisit this soon. This article also has a nice little [installation video](https://www.youtube.com/watch?v=Rgpfk6eYxJA), as well.
* [Article by Gilbert Tanner](https://gilberttanner.com/blog/installing-the-tensorflow-object-detection-api) or his [3-part YouTube video tutorial](https://www.youtube.com/watch?v=wdufj-pjE5c&feature=youtu.be): I read this earlier than most others, so I have to assume there was some trouble while following his installation instructions. But, I like how he explains things, and he also has a good explanation of creating your own data and training in video #3.
* [InsightBot Article](http://www.insightsbot.com/blog/1KYUd3/tensorflow-object-detection-api-windows-install-guide): Another brief walkthrough of protobuf (this one uses 3.4.0), and puts a lower bound on tf 1.4.* or higher. Links to a [follow-up article](http://www.insightsbot.com/blog/womeQ/tensorflow-object-detection-tutorial-on-images) on using the Object Detection API, but I didn't end up following this.
* [Beyond Data Science Article](https://towardsdatascience.com/custom-object-detection-for-non-data-scientists-70325fef2dbb): The installation instructions didn't work for me, but the training section seems worth checking out.
* [Becoming Human Article](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73): This one has a flowchart, which is pretty cool. Talks about partitioning the training dataset, I think, but I've been skimming so many articles for days so I'm adding this on the list of things to go back and read more fully.
* [Wiki Article from the OpenCV Github](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API): Only skimmed this one, too. I'm really just trying to close all my tabs now that I've finished an MVP of this (while keeping a record of the worthwhile-looking ones!)

---

### Next Steps
* Followed Gilbert's guide for taking, photos, organizing, labeling, making xmls, csvs, the pbtxt, etc. This works up till trying to run the train.py as he describes, because I'm on tf 1.5 and not 1.9
* From here, switched over to Edje Electronics' guide and downloaded the tf 1.5 model folder that they link
* I already had my PYTHONPATH configured as they say to--maybe I should add this as an instruction earlier in this guide TODO
* Compiled protobufs:
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```
* Things weren't working, removed pip's brotobuf, then conda-installed protobuf 3.6.0
* I'd run the python setup.py build/install a couple times while messing with this, idk if it'll be relevant later
* I also ended up taking away a few of the protoc commands for files that no longer seemed to exist
* By this point, stopped getting the error where it couldn't find preprocessor pb2 or whatever it specifically was
* Instead, we get an issue that seems to be related to matplotlib and qt compatibility - is this because we're on matplotlib 2.2.2?
* conda-uninstalled matplotlib, then conda-installed it again (this time it installed as 3.0.0). Now when trying to run `python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training\faster_rcnn_inception_v2_pets.config` we get a different error 
* As per [this](https://github.com/tensorflow/models/issues/5451), I took out the line about object detection and now train.py is running!

