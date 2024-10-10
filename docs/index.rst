.. PyTorch Warmup documentation master file, created by
   sphinx-quickstart on Thu Oct 31 14:00:43 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTorch Warmup's documentation!
==========================================

This library contains PyTorch implementations of the warmup schedules described in
`On the adequacy of untuned warmup for adaptive optimization
<https://arxiv.org/abs/1910.04209>`_.

.. image:: https://github.com/Tony-Y/pytorch_warmup/raw/master/examples/plots/figs/warmup_schedule.png
   :alt: Warmup schedule
   :width: 400
   :align: center

.. image:: https://github.com/Tony-Y/pytorch_warmup/workflows/Python%20package/badge.svg
   :alt: Python package
   :target: https://github.com/Tony-Y/pytorch_warmup/

.. image:: https://img.shields.io/pypi/v/pytorch-warmup.svg
   :alt: PyPI version shields.io
   :target: https://pypi.python.org/pypi/pytorch-warmup/

.. image:: https://img.shields.io/pypi/l/pytorch-warmup.svg
   :alt: PyPI license
   :target: https://github.com/Tony-Y/pytorch_warmup/blob/master/LICENSE

.. image:: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue
   :alt: Python versions
   :target: https://www.python.org

Installation
------------

Make sure you have Python 3.7+ and PyTorch 1.1+ or 2.x. Then, install the latest version from the Python Package Index:

.. code-block:: shell

   pip install -U pytorch_warmup

Examples
--------

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Open In Colab
   :target: https://colab.research.google.com/github/Tony-Y/colab-notebooks/blob/master/PyTorch_Warmup_Approach1_chaining.ipynb
   
* `EMNIST <https://github.com/Tony-Y/pytorch_warmup/tree/master/examples/emnist>`_ -
  A sample script to train a CNN model on the EMNIST dataset using the Adam algorithm with a warmup.

* `Plots <https://github.com/Tony-Y/pytorch_warmup/tree/master/examples/plots>`_ -
  A script to plot effective warmup periods as a function of :math:`\beta_{2}`, and warmup schedules over time.

Usage
-----

When the learning rate schedule uses the global iteration number, the untuned linear warmup can be used
together with :class:`Adam` or its variant (:class:`AdamW`, :class:`NAdam`, etc.) as follows:

.. code-block:: python

   import torch
   import pytorch_warmup as warmup

   optimizer = torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
      # This sample code uses the AdamW optimizer.
   num_steps = len(dataloader) * num_epochs
   lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
      # The LR schedule initialization resets the initial LR of the optimizer.
   warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
      # The warmup schedule initialization dampens the initial LR of the optimizer.
   for epoch in range(1,num_epochs+1):
      for batch in dataloader:
         optimizer.zero_grad()
         loss = ...
         loss.backward()
         optimizer.step()
         with warmup_scheduler.dampening():
               lr_scheduler.step()

Note that the warmup schedule must not be initialized before the initialization of the learning rate schedule.
Other approaches can be found in `README <https://github.com/Tony-Y/pytorch_warmup?tab=readme-ov-file#usage>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   manual_warmup
   untuned_warmup
   radam_warmup


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
