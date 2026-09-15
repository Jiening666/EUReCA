import struct
import numpy as np
import warnings

def read_xim(filename, read_pixel_data=True):
    """
    Reads XIM image files from Varian TrueBeam accelerators.

    This is a Python port of the MATLAB function ReadXim.m by 
    Fredrik Nordström and C. Pelizzari.

    Args:
        filename (str): The full path to the .xim image file.
        read_pixel_data (bool): Determines whether pixel data should be decoded.
                                True: decode (default)
                                False: skip

    Returns:
        dict: A dictionary containing the XIM image information
              and pixel data (if read_pixel_data is True).
    """
    
    # The 'img' dictionary will store all the data, similar to the MATLAB struct
    img = {}
    img['file_name'] = filename

    # Use '<' for little-endian, which is standard for most systems (like x86).
    # The MATLAB '*' prefix means "native," which is usually little-endian.
    
    with open(filename, 'rb') as f:
        # --- Decode header ---
        img['file_format_identifier'] = f.read(8).decode('ascii')
        
        # Read multiple 32-bit integers at once
        header_data = struct.unpack('<iiiiii', f.read(6 * 4))
        img['file_format_version'] = header_data[0]
        img['image_width'] = header_data[1]
        img['image_height'] = header_data[2]
        img['bits_per_pixel'] = header_data[3]
        img['bytes_per_pixel'] = header_data[4]
        img['compression_indicator'] = header_data[5]

        # --- Decode pixel data ---
        if img['compression_indicator'] == 1:
            # --- Compressed Image ---
            # Read lookup_table_size (int32, little endian)
            lookup_table_size = struct.unpack('<i', f.read(4))[0]

            # Read raw bytes: each byte encodes 4 values (2 bits each)
            raw_bytes = np.fromfile(f, dtype=np.uint8, count=lookup_table_size)

            # Extract 4 two-bit values per byte
            # Equivalent to MATLAB's 'ubit2'
            vals = np.empty(lookup_table_size * 4, dtype=np.uint8)
            vals[0::4] = (raw_bytes >> 0) & 0x03  # bits 7–6
            vals[1::4] = (raw_bytes >> 2) & 0x03  # bits 5–4
            vals[2::4] = (raw_bytes >> 4) & 0x03  # bits 3–2
            vals[3::4] = (raw_bytes >> 6) & 0x03  # bits 1–0
            img['lookup_table'] = vals

            img['compressed_pixel_buffer_size'] = struct.unpack('<i', f.read(4))[0]

            if read_pixel_data:
                # The MATLAB code has a fast C (MEX) version ('myreadxim')
                # and a slow MATLAB version. We implement the slow version.
                #warnings.warn(
                #    "Implementing slow Python decompression logic (port of "
                #    "MATLAB 'else' block). The 'myreadxim' C function "
                #    "is not available."
                #)

                pixel_count = img['image_width'] * img['image_height']
                pixel_data = np.zeros(pixel_count, dtype=np.int32)

                # Read first (width + 1) pixels
                # (MATLAB 1-indexing: 1:img.image_width+1)
                init_read_count = img['image_width'] + 1
                init_data = np.fromfile(f, dtype=np.int32, count=init_read_count)
                pixel_data[0:init_read_count] = init_data

                lookup_table_pos = 0
                
                # Loop for the rest of the pixels
                # (MATLAB 1-indexing: (img.image_width+2):(img.image_width*img.image_height))
                for image_pos in range(init_read_count, pixel_count):
                    lut_val = img['lookup_table'][lookup_table_pos]

                    if lut_val == 0:
                        diff = struct.unpack('<b', f.read(1))[0]  # int8
                    elif lut_val == 1:
                        diff = struct.unpack('<h', f.read(2))[0]  # int16
                    else:
                        diff = struct.unpack('<i', f.read(4))[0]  # int32

                    # Decompression algorithm (adjusting for 0-indexing)
                    
                    diff_32 = np.int32(diff)
                    
                    # The other elements are already np.int32
                    p1 = pixel_data[image_pos - 1]
                    p2 = pixel_data[image_pos - img['image_width']]
                    p3 = pixel_data[image_pos - img['image_width'] - 1]

                    # This calculation will now correctly overflow and wrap.
                    # We also silence the (now expected) overflow warning.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        pixel_data[image_pos] = diff_32 + p1 + p2 - p3
                    
                    lookup_table_pos += 1

                # Reshape and cast. XIM 16-bit detector data is UNSIGNED;
                # casting to signed int16 wraps values > 32767 (e.g. HighGain
                # bowtie peaks ~34289) to negative -> corrupted flat-field.
                if img['bytes_per_pixel'] == 2:
                    pixel_data = pixel_data.astype(np.uint16)
                
                # MATLAB's reshape(..., w, h)' is equivalent to 
                # NumPy's reshape(..., h, w)
                img['pixel_data'] = pixel_data.reshape(
                    (img['image_height'], img['image_width'])
                )

            else:
                # Skip pixel data
                f.seek(img['compressed_pixel_buffer_size'], 1)  # 1 = seek from current pos

            img['uncompressed_pixel_buffer_size'] = struct.unpack('<i', f.read(4))[0]
        
        else:
            # --- Uncompressed Image ---
            img['uncompressed_pixel_buffer_size'] = struct.unpack('<i', f.read(4))[0]
            
            if read_pixel_data:
                if img['bytes_per_pixel'] == 1:
                    dtype = np.int8
                    count = img['uncompressed_pixel_buffer_size']
                elif img['bytes_per_pixel'] == 2:
                    dtype = np.uint16   # XIM 16-bit detector data is unsigned
                    count = img['uncompressed_pixel_buffer_size'] // 2
                else:
                    dtype = np.int32
                    count = img['uncompressed_pixel_buffer_size'] // 4

                pixel_data = np.fromfile(f, dtype=dtype, count=count)
                img['pixel_data'] = pixel_data.reshape(
                    (img['image_height'], img['image_width'])
                )
            else:
                # Skip pixel data
                f.seek(img['uncompressed_pixel_buffer_size'], 1)

        # --- Decode histogram ---
        num_bins = struct.unpack('<i', f.read(4))[0]
        if num_bins > 0:
            img['histogram'] = {
                'number_of_bins': num_bins,
                'data': np.fromfile(f, dtype=np.int32, count=num_bins)
            }

        # --- Decode properties ---
        num_props = struct.unpack('<i', f.read(4))[0]
        properties = {}
        if num_props > 0:
            for _ in range(num_props):
                try:
                    name_len = struct.unpack('<i', f.read(4))[0]
                    name = f.read(name_len).decode('ascii')
                    prop_type = struct.unpack('<i', f.read(4))[0]

                    value = None
                    if prop_type == 0:   # int32
                        value = struct.unpack('<i', f.read(4))[0]
                    elif prop_type == 1: # double
                        value = struct.unpack('<d', f.read(8))[0]
                    elif prop_type == 2: # string
                        val_len = struct.unpack('<i', f.read(4))[0]
                        value = f.read(val_len).decode('ascii')
                    elif prop_type == 4: # array of double
                        val_len = struct.unpack('<i', f.read(4))[0]
                        count = val_len // 8
                        value = np.fromfile(f, dtype=np.float64, count=count)
                    elif prop_type == 5: # array of int32
                        val_len = struct.unpack('<i', f.read(4))[0]
                        count = val_len // 4
                        value = np.fromfile(f, dtype=np.int32, count=count)
                    else:
                        print(
                            f"Warning: Property '{name}' has unsupported "
                            f"type {prop_type}. "
                            "Aborting property decoding."
                        )
                        break  # Exit the property reading loop
                    
                    if value is not None:
                        properties[name] = value
                
                except struct.error:
                    print("Error reading properties. File may be truncated.")
                    break

        # Sort properties by key (name) to match MATLAB output
        img['properties'] = {
            k: properties[k] for k in sorted(properties.keys())
        }

    return img