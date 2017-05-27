Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Anaconda2\lib\site-packages\spyderlib\widgets\externalshell\sitecustomize.py", line 699, in runfile
    execfile(filename, namespace)
  File "C:\Anaconda2\lib\site-packages\spyderlib\widgets\externalshell\sitecustomize.py", line 74, in execfile
    exec(compile(scripttext, filename, 'exec'), glob, loc)
  File "C:/Users/Danney/Documents/GitHub/OCR/main.py", line 91, in <module>
    main()
  File "C:/Users/Danney/Documents/GitHub/OCR/main.py", line 75, in main
    lines = gc.get_all(im, params)
  File "get_chars.py", line 210, in get_all
    im_words = get_words_from_line(params, im_line)
  File "get_chars.py", line 133, in get_words_from_line
    regions = regionprops(label_image)
  File "C:\Anaconda2\lib\site-packages\scikit-image\skimage\measure\_regionprops.py", line 536, in regionprops
    raise TypeError('Only 2-D and 3-D images supported.')
TypeError: Only 2-D and 3-D images supported.