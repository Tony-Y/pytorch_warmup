#!/bin/bash -ex
#
# In the project root directry, run this script to build the HTML document:
#   >> ./docs/build.sh
#

export SMV_BRANCH_WHITELIST="^$(git branch --show-current)$"

SOURCEDIR=docs
BUILDDIR=build/html

sphinx-multiversion $SOURCEDIR $BUILDDIR
cp $SOURCEDIR/gh-pages-redirect.html $BUILDDIR/index.html
touch $BUILDDIR/.nojekyll
