# BGWpy Tutorial #

This repo contains a tutorial for running BGWpy as a Jupyter Notebooky, as well
as a simple toolchain to generate a static website based on the notebook.

We have chosen to use `mkdocs` and the `Material` theme to maximize
compatibility with the main BerkeleyGW online documentation.

# Requirements #

The following are needed to generate the static website for this tutorial:

- mkdocs
- mkdocs-material
- nbconvert (part of the Jupyter ecosystem)
- GNU Make

The requirements for running the tutorial are contained in the tutorial.

# Installation #

Type `make` in the root directory.  The website will be generated into the
`site` subdirectory.

# Notes #

GitHub Pages requires that personal pages be hosted from the root folder of the
repo, with no option to use the `gh-deploy` branch or a subdirectory (that's for
paying customers).  To get around this, I've added the `site` subdirectory to
version control and created a simple `index.html` file in the root folder that
redirects to the `site` subdirectory

If you are viewing this website from your local filesystem, you will need to set
`use_directory_urls: false` in the <mkdocs.yml> file to properly view all
webpages other than the main page.
