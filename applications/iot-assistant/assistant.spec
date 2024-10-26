import os

# Get absolute paths
config_path = os.path.abspath('config.yml')
interaces_path = os.path.abspath('interaces')

block_cipher = None

a = Analysis(
    ['pico.py'],
    pathex=[os.path.abspath('.')],  # Absolute path to the current directory
    binaries=[],
    datas=[
        (config_path, '.'),  # Include the config file with absolute path
        (interaces_path, 'interaces'),  # Include the entire interaces directory with absolute path
    ],
    hiddenimports=[
        'interaces.email_interface',
        'interaces.spotify_interface',
        'interaces.twilio_sms_interface',
        'interaces.weather_interface',
        'interaces.whatsapp_interface',
        'interaces.google_interface',
        'interaces.stock_interface',
        'text_to_speech',
        'scipy.special._ufuncs',
        'scipy.special._ufuncs_cxx',
        'scipy.special._ellip_harm',
        'scipy.special._ellip_harm_2',
        'scipy.special.specfun',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'scipy._lib.messagestream',
        'scipy.special._cdflib'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False if you don't want a console window
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pico'
)
