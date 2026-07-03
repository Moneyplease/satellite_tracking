import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from photutils.background import Background2D, MedianBackground
from skimage import measure, exposure

def extract_lines_from_fits(file_path):
    with fits.open(file_path) as hdulist:
        data = hdulist[0].data.astype(float)
        header = hdulist[0].header
        
        exptime = float(header.get('EXPTIME', 1.0))
        obs_time = header.get('DATE-OBS', header.get('TIME-OBS', 'Unknown Time'))
    
        bkg = Background2D(data, (64, 64), filter_size=(3, 3), bkg_estimator=MedianBackground())
        sub_image = data - bkg.background
        rms = float(np.median(bkg.background_rms))
        
    binary_mask = (sub_image > (3.5 * rms)).astype(np.uint8)
    label_image = measure.label(binary_mask, connectivity=2)
    properties = measure.regionprops(label_image)
    
    lines = []
    for prop in properties:
        if prop.area < 5 or prop.perimeter == 0:
            continue
        
        eccentricity = prop.eccentricity
        line_length = prop.axis_major_length
        
        if eccentricity >= 0.95 and line_length >= 10:
            lines.append({
                'id': prop.label,
                'centroid': prop.centroid,       
                'orientation': prop.orientation, 
                'length': line_length,
                'bbox': prop.bbox
            })
    return lines, exptime, obs_time

def plate_solve_from_image(fits_path, api_key, solve_timeout=600, force_upload=True):

    from astroquery.astrometry_net import AstrometryNet
    from astropy.wcs import WCS as _WCS
    ast = AstrometryNet()
    ast.api_key = api_key
    print(f"Uploading image to astrometry.net")
    try:
        hdr = ast.solve_from_image(fits_path, force_image_upload=force_upload,
                                   solve_timeout=solve_timeout, publicly_visible='n')
    except TimeoutError:
        print("Plate solve timed out."); return None
    except Exception as exc:
        print(f"Plate solve error: {exc}"); return None
    if not hdr:
        print("No solution."); return None
    print("Plate solve succeeded (WCS recovered).")
    return _WCS(hdr)

def visualize_tracking_no_arrows(file1, file2, matches, exptime, time1, time2, wcs2=None):
    """ฟังก์ชันสำหรับวาดภาพเปรียบเทียบ Frame 1 และ Frame 2 พร้อมแสดงค่าความเร็ว"""
    with fits.open(file1) as h1, fits.open(file2) as h2:
        img1 = h1[0].data.astype(float)
        img2 = h2[0].data.astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    vmin1, vmax1 = np.percentile(img1, [1, 99])
    vmin2, vmax2 = np.percentile(img2, [1, 99])

    ax1.imshow(img1, cmap='gray', origin='lower', vmin=vmin1, vmax=vmax1)
    ax1.set_title(f"Frame 1: {time1}")
    
    ax2.imshow(img2, cmap='gray', origin='lower', vmin=vmin2, vmax=vmax2)
    ax2.set_title(f"Frame 2: {time2} (With Velocity)")

    for l1, l2 in matches:  
        y1, x1 = l1['centroid']
        y2, x2 = l2['centroid']
        
        label_text = f"Match:{l2['id']}"

        if wcs2 is not None:
            ra2, dec2 = wcs2.pixel_to_world_values(x2, y2)
            ra1, dec1 = wcs2.pixel_to_world_values(x1, y1)
            distance_deg = np.sqrt((ra2 - ra1)**2 + (dec2 - dec1)**2)
            velocity_deg = distance_deg / exptime
            label_text += f"\n{velocity_deg:.4f} °/s"
        
        ax1.plot(x1, y1, 'ro', markersize=5) 
        ax1.text(x1 + 10, y1 + 10, f"ID:{l1['id']}", color='lime', fontsize=10, weight='bold')
        
        ax2.plot(x2, y2, 'go', markersize=5) 
        ax2.text(x2 + 10, y2 + 10, label_text, color='lime', fontsize=9, weight='bold')
        
        ax2.plot(x1, y1, 'rx', alpha=0.4)

    plt.tight_layout()
    output_vis = "tracking_comparison_with_wcs.png"
    plt.savefig(output_vis, dpi=200)
    print(f"\n[Visual] เซฟภาพเปรียบเทียบเรียบร้อย: '{output_vis}'")
    plt.show()

def visualize_processing_steps(file_path):

    with fits.open(file_path) as hdulist:
        raw_data = hdulist[0].data.astype(float)
        
    vmin, vmax = np.percentile(raw_data, [1, 99])
    
    
    bkg = Background2D(raw_data, (64, 64), filter_size=(3, 3), bkg_estimator=MedianBackground())
    sub_image = raw_data - bkg.background
    rms = float(np.median(bkg.background_rms))
    
    img_normalized = exposure.rescale_intensity(sub_image, out_range=(0, 1))
    img_hist_eq = exposure.equalize_hist(img_normalized)
    binary_mask = (sub_image > (5 * rms)).astype(np.uint8)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.imshow(raw_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax1.set_title("1. Raw Image (Before Clean)")
    ax2.imshow(img_hist_eq, cmap='gray', origin='lower')
    ax2.set_title("2. Histogram Equalized")
    ax3.imshow(binary_mask, cmap='gray', origin='lower')
    ax3.set_title("3. Thresholded (Binary Mask)")
    plt.tight_layout()
    plt.show()

def process_and_track_frames(file1, file2, astrometry_api_key=None):
    wcs2 = None
    if astrometry_api_key:
        print("--- Starting Astrometry Plate Solving ---")
        wcs2 = plate_solve_from_image(file2, astrometry_api_key)
        print("-----------------------------------------\n")

    print("Extracting features from Frame 1...")
    lines_frame1, exptime1, time1 = extract_lines_from_fits(file1)
    print(f"  Found {len(lines_frame1)} lines. | Time: {time1} | Exposure: {exptime1}s")

    print("Extracting features from Frame 2...")
    lines_frame2, time2 = extract_lines_from_fits(file2)
    print(f"  Found {len(lines_frame2)} lines. | Time: {time2}\n")

    print("--- Tracking & Velocity Analysis ---")

    ANGLE_THRESHOLD_DEG = 10.0   
    MAX_DISTANCE = 150.0         
    matched_pairs = []  

    for l1 in lines_frame1:
        y1, x1 = l1['centroid']
        ang1_deg = np.degrees(l1['orientation'])
        best_match = None
        min_dist = float('inf')
        
        for l2 in lines_frame2:
            y2, x2 = l2['centroid']
            ang2_deg = np.degrees(l2['orientation'])
            
            angle_diff = abs(ang1_deg - ang2_deg)
            if angle_diff > 90:  
                angle_diff = 180 - angle_diff
                
            if angle_diff > ANGLE_THRESHOLD_DEG:
                continue 
                
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance > MAX_DISTANCE:
                continue 
                
            if distance < min_dist:
                min_dist = distance
                best_match = l2

        if best_match is not None:
            matched_pairs.append((l1, best_match)) 
            y2, x2 = best_match['centroid']
            
            dy = y2 - y1
            dx = x2 - x1
            dir_y = "UP" if dy > 0 else "DOWN"
            dir_x = "RIGHT" if dx > 0 else "LEFT"
            
            print(f" Match Found: Frame1[ID {l1['id']:2d}], Frame2[ID {best_match['id']:2d}]")
            print(f" Time1: {time1}, Time2: {time2}, t_diff: {exptime1:.1f}s")
            print(f" Direction: {dir_y}, {dir_x}")
            
            if wcs2 is not None:
                ra1, dec1 = wcs2.pixel_to_world_values(x1, y1)
                ra2, dec2 = wcs2.pixel_to_world_values(x2, y2)
                distance_deg = np.sqrt((ra2 - ra1) ** 2 + (dec2 - dec1) ** 2)
                velocity_deg = distance_deg / exptime1
                print(f" RA={ra2:.5f}°, Dec={dec2:.5f}°")
                print(f" Angular Velocity: {velocity_deg:.5f} degrees/second")
                
            print(f" Angle Difference: {angle_diff:.2f}°\n")
        else:
            print(f"No Object Matched")

    if len(matched_pairs) > 0:
        visualize_tracking_no_arrows(file1, file2, matched_pairs, exptime1, time1, time2, wcs2)
    else:
        print(f"No Object Matched")
    
if __name__ == "__main__":
    path1 = "/Users/none/internship/fit_files/CZ-3B DEB-0051Emty.fit"
    path2 = "/Users/none/internship/fit_files/CZ-3B DEB-0052Emty.fit" 
    
    MY_API_KEY = '' #jsijqcvqthalyjjj
    
    visualize_processing_steps(path1)
    
    process_and_track_frames(path1, path2, astrometry_api_key=MY_API_KEY)