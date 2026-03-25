# Pyodide Built-in Packages

These packages are pre-built for Pyodide and available in WASM environments.
Any package **not** on this list must have a pure Python wheel on PyPI to work.

> **Note:** This list was snapshotted on 2026-02-26 from Pyodide's docs.
> For the latest list, check https://pyodide.org/en/stable/usage/packages-in-pyodide.html

affine, aiohappyeyeballs, aiohttp, aiosignal, altair, annotated-types, anyio,
apsw, argon2-cffi, argon2-cffi-bindings, asciitree, astropy, astropy_iers_data,
asttokens, async-timeout, atomicwrites, attrs, audioop-lts, autograd,
awkward-cpp, b2d, bcrypt, beautifulsoup4, bilby.cython, biopython, bitarray,
bitstring, bleach, blosc2, bokeh, boost-histogram, Bottleneck, brotli,
cachetools, Cartopy, casadi, cbor-diag, certifi, cffi, cffi_example, cftime,
charset-normalizer, clarabel, click, cligj, clingo, cloudpickle, cmyt, cobs,
colorspacious, contourpy, coolprop, coverage, cramjam, crc32c, cryptography,
css-inline, cssselect, cvxpy-base, cycler, cysignals, cytoolz, decorator,
demes, deprecation, diskcache, distlib, distro, docutils, donfig,
ewah_bool_utils, exceptiongroup, executing, fastapi, fastcan, fastparquet,
fiona, fonttools, freesasa, frozenlist, fsspec, future, galpy, geopandas,
gmpy2, google-crc32c, gsw, h11, h3, h5py, healpy, highspy, html5lib, httpcore,
httpx, idna, igraph, imageio, imgui-bundle, iminuit, iniconfig, inspice,
ipython, jedi, Jinja2, jiter, joblib, jsonpatch, jsonpointer, jsonschema,
jsonschema_specifications, kiwisolver, lakers-python, lazy_loader,
lazy-object-proxy, libcst, lightgbm, logbook, lxml, lz4, MarkupSafe,
matplotlib, matplotlib-inline, memory-allocator, micropip, ml_dtypes, mmh3,
more-itertools, mpmath, msgpack, msgspec, msprime, multidict, munch, mypy,
narwhals, ndindex, netcdf4, networkx, newick, nh3, nlopt, nltk, numcodecs,
numpy, openai, opencv-python, optlang, orjson, packaging, pandas, parso, patsy,
pcodec, peewee, pi-heif, Pillow, pillow-heif, pkgconfig, platformdirs, pluggy,
ply, pplpy, primecountpy, prompt_toolkit, propcache, protobuf, pure-eval, py,
pyarrow, pycdfpp, pyclipper, pycparser, pycryptodome, pydantic, pydantic_core,
pyerfa, pygame-ce, Pygments, pyheif, pyiceberg, pyinstrument, pylimer-tools,
PyMuPDF, pynacl, pyodide-http, pyodide-unix-timezones, pyparsing, pyproj,
pyrodigal, pyrsistent, pysam, pyshp, pytaglib, pytest, pytest-asyncio,
pytest-benchmark, pytest_httpx, python-calamine, python-dateutil, python-flint,
python-magic, python-sat, python-solvespace, pytz, pywavelets, pyxel, pyxirr,
pyyaml, rasterio, rateslib, rebound, reboundx, referencing, regex, requests,
retrying, rich, river, RobotRaconteur, rpds-py, ruamel.yaml, rustworkx,
scikit-image, scikit-learn, scipy, screed, setuptools, shapely, simplejson,
sisl, six, smart-open, sniffio, sortedcontainers, soundfile, soupsieve,
sourmash, soxr, sparseqr, sqlalchemy, stack-data, starlette, statsmodels,
strictyaml, svgwrite, swiglpk, sympy, tblib, termcolor, texttable,
texture2ddecoder, threadpoolctl, tiktoken, tomli, tomli-w, toolz, tqdm,
traitlets, traits, tree-sitter, tree-sitter-go, tree-sitter-java,
tree-sitter-python, tskit, typing-extensions, typing-inspection, tzdata, ujson,
uncertainties, unyt, urllib3, vega-datasets, vrplib, wcwidth, webencodings,
wordcloud, wrapt, xarray, xgboost, xlrd, xxhash, xyzservices, yarl, yt, zengl,
zfpy, zstandard

## Also available (part of Pyodide runtime or marimo WASM)

- marimo
- duckdb
- polars
- micropip (for installing additional pure-Python packages at runtime)

## Common third-party packages that do NOT work in WASM

These popular packages have C/native extensions not built for Pyodide:

| Package | Why | Alternative |
|---|---|---|
| torch / pytorch | C++/CUDA extensions | None for WASM |
| tensorflow | C++ extensions | None for WASM |
| jax / jaxlib | C++ extensions | None for WASM |
| psycopg2 | Requires libpq | `psycopg[binary]` or use `duckdb` |
| mysqlclient | Requires libmysqlclient | `pymysql` (pure Python) |
| uvloop | Requires libuv | `asyncio` (default loop) |
| grpcio | C extensions | `grpclib` (pure Python) |
| psutil | OS-level syscalls | None for WASM |
| gevent | C extensions | `asyncio` |
| celery | Requires message broker | Not applicable in browser |
