MVP: Generates cross stitch patterns from simple images for basic Aida 14ct fabric (10x10 grid)

Function: Feeds algorithm an image, specifies number of palette colors, extracts color palette from image for number of colors specified and matches them to a thread library based on DMC colors, generates cross stitch grid and key, reveals image with thread colors applied

Current Issues:  does not generate the pattern grid correctly. working on alternate approaches.

TODO: implement testing, improve modularity 

Additional Features:  once basic generaton of pattern is established, start iterating on additional functionality, like:
  - increasingly complex images with more elaborate color schemes and greate resolution (pixels per inch) 
  - incorporate embroidery stitching techniques for better texturing and detail (may require additional modeling)
  - expanding thread databse for increased options and more like renderings to original image
  - additional options for pattern generation like higher thread count Aida and alternate stitching techniques
  - Add React UI
