import streamlit as st
import os
import requests
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import tempfile
from geopy.geocoders import Nominatim
import time
import xml.etree.ElementTree as ET
import gc
from threading import Lock

# Extreme memory conservation settings
plt.ioff()  # Turn off interactive plotting
np.seterr(all='ignore')  # Suppress numpy warnings

# Set page config
st.set_page_config(
    page_title="GOES Satellite Viewer", 
    page_icon="🛰️",
    layout="wide"
)

# Resource management
processing_lock = Lock()

# Satellite configurations
SATELLITES = {
    'GOES-16': {
        'longitude': -75.2,
        'start_date': datetime(2017, 4, 1, 0, 0),
        'end_date': datetime(2024, 10, 10, 20, 0),
        'bucket': 'noaa-goes16',
        'identifier': 'G16'
    },
    'GOES-17': {
        'longitude': -137.0,
        'start_date': datetime(2018, 8, 28, 0, 0),
        'end_date': datetime(2022, 7, 27, 23, 59),
        'bucket': 'noaa-goes17',
        'identifier': 'G17'
    },
    'GOES-18': {
        'longitude': -137.0,
        'start_date': datetime(2022, 7, 28, 0, 0),
        'end_date': datetime(2030, 12, 31, 23, 59),
        'bucket': 'noaa-goes18',
        'identifier': 'G18'
    },
    'GOES-19': {
        'longitude': -75.2,
        'start_date': datetime(2024, 10, 10, 20, 0),
        'end_date': datetime(2030, 12, 31, 23, 59),
        'bucket': 'noaa-goes19',
        'identifier': 'G19'
    }
}

def force_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    time.sleep(0.1)

def rbtop3():
    """Minimal colormap"""
    colors = ["#000000", "#05fcfe", "#010071", "#00fe24", "#fbff2d", "#fd1917", "#eb6fc0", "#330f2f"]
    return mcolors.LinearSegmentedColormap.from_list("", colors).reversed()

def get_coordinates(location_str):
    """Lightweight geocoding"""
    try:
        geolocator = Nominatim(user_agent="goes_sat_viewer", timeout=3)
        time.sleep(1)
        location = geolocator.geocode(location_str, timeout=3)
        if location:
            return location.latitude, location.longitude
        return None, None
    except:
        return None, None

def select_best_satellite(lon, lat, requested_date):
    """Select optimal satellite"""
    best_satellite = None
    min_distance = float('inf')
    
    for sat_name, sat_info in SATELLITES.items():
        if sat_info['start_date'] <= requested_date <= sat_info['end_date']:
            distance = abs(lon - sat_info['longitude'])
            if distance > 180:
                distance = 360 - distance
            if distance < min_distance:
                min_distance = distance
                best_satellite = sat_name
    
    return best_satellite, None if best_satellite else "No satellite available"

def list_files_minimal(bucket_url):
    """Minimal file listing with timeout"""
    try:
        response = requests.get(bucket_url, timeout=10)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            files = [contents.find('{http://s3.amazonaws.com/doc/2006-03-01/}Key').text 
                    for contents in root.findall('{http://s3.amazonaws.com/doc/2006-03-01/}Contents')]
            return files
        return []
    except:
        return []

def download_minimal(satellite_name, year, day_of_year, hour, progress_bar):
    """Ultra-minimal download with aggressive chunking"""
    sat_info = SATELLITES[satellite_name]
    bucket = sat_info['bucket']
    identifier = sat_info['identifier']
    
    # Validate date
    requested_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour)
    if not (sat_info['start_date'] <= requested_date <= sat_info['end_date']):
        raise Exception(f"Date outside {satellite_name} coverage")
    
    progress_bar.progress(10, "Finding satellite file...")
    
    bucket_url = f"https://{bucket}.s3.amazonaws.com/?prefix=ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/"
    files = list_files_minimal(bucket_url)
    
    # Find file pattern
    if satellite_name == 'GOES-16' and requested_date < datetime(2019, 4, 2, 16):
        pattern = f"OR_ABI-L1b-RadF-M3C13_{identifier}_s{year}{day_of_year:03d}{hour:02d}00"
    else:
        pattern = f"OR_ABI-L1b-RadF-M6C13_{identifier}_s{year}{day_of_year:03d}{hour:02d}00"
    
    filename = None
    for file in files:
        if file.startswith(f"ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/{pattern}"):
            filename = file.split('/')[-1]
            break
    
    if not filename:
        raise Exception(f"No {satellite_name} file found")
    
    progress_bar.progress(30, "Downloading satellite data...")
    
    # Ultra-chunked download
    download_url = f"https://{bucket}.s3.amazonaws.com/ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/{filename}"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
        save_path = tmp_file.name
    
    try:
        response = requests.get(download_url, timeout=60, stream=True)
        if response.status_code == 200:
            downloaded = 0
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=4096):  # Very small chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (1024*1024) == 0:  # Every MB
                            progress_bar.progress(min(75, 30 + int(45 * downloaded / (50*1024*1024))), 
                                                f"Downloaded: {downloaded/(1024*1024):.1f}MB")
                            time.sleep(0.05)  # CPU break
            return save_path, filename
        else:
            raise Exception(f"Download failed: {response.status_code}")
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise

def process_minimal_region(file_path, filename, center_coord, progress_bar):
    """Process only a tiny region to minimize memory usage"""
    
    progress_bar.progress(80, "Processing satellite data...")
    
    center_lon, center_lat = center_coord
    
    # Very small region to minimize memory
    region_size = 6  # Only 12x12 degree region
    lon_min, lon_max = center_lon - region_size, center_lon + region_size
    lat_min, lat_max = center_lat - region_size, center_lat + region_size
    
    try:
        with nc.Dataset(file_path, 'r') as ds:
            
            # Get a small subset of coordinates for coverage check
            x_full = ds.variables['x'][:]
            y_full = ds.variables['y'][:]
            
            # Heavy subsampling for coordinate check
            subsample = 20  # Use every 20th pixel for coverage check
            x_check = x_full[::subsample]
            y_check = y_full[::subsample]
            
            # Compute coordinates for coverage check only
            proj_info = ds.variables['goes_imager_projection']
            lon_origin = proj_info.longitude_of_projection_origin
            H = proj_info.perspective_point_height + proj_info.semi_major_axis
            r_eq = proj_info.semi_major_axis
            r_pol = proj_info.semi_minor_axis

            X_check, Y_check = np.meshgrid(x_check, y_check)
            
            # Minimal coordinate computation for coverage check
            lambda_0 = np.deg2rad(lon_origin)
            a_var = np.sin(X_check) ** 2 + (np.cos(X_check) ** 2 * (np.cos(Y_check) ** 2 + ((r_eq ** 2 / r_pol ** 2) * np.sin(Y_check) ** 2)))
            b_var = -2 * H * np.cos(X_check) * np.cos(Y_check)
            c_var = H ** 2 - r_eq ** 2
            
            discriminant = np.maximum(b_var ** 2 - 4 * a_var * c_var, 0)
            r_s = (-b_var - np.sqrt(discriminant)) / (2 * a_var)
            s_x = r_s * np.cos(X_check) * np.cos(Y_check)
            s_y = -r_s * np.sin(X_check)
            s_z = r_s * np.cos(X_check) * np.sin(Y_check)
            
            denominator = np.sqrt((H - s_x) ** 2 + s_y ** 2)
            denominator = np.where(denominator == 0, 1e-10, denominator)
            
            lat_check = np.rad2deg(np.arctan((r_eq ** 2 / r_pol ** 2) * (s_z / denominator)))
            lon_check = np.rad2deg(lambda_0 - np.arctan(s_y / (H - s_x)))
            
            # Check if location is covered
            coverage_mask = ((lon_check >= lon_min) & (lon_check <= lon_max) & 
                           (lat_check >= lat_min) & (lat_check <= lat_max))
            
            if not np.any(coverage_mask):
                raise Exception("Location not covered by satellite")
            
            # Clean up check arrays
            del X_check, Y_check, lat_check, lon_check, coverage_mask
            del a_var, b_var, c_var, discriminant, r_s, s_x, s_y, s_z, denominator
            force_cleanup()
            
            progress_bar.progress(85, "Loading data region...")
            
            # Find approximate indices for our region
            # Heavy subsampling for actual data processing
            data_subsample = 8  # Use every 8th pixel
            
            # Load heavily subsampled data and coordinates
            data = ds.variables['Rad'][::data_subsample, ::data_subsample].squeeze().astype(np.float32)
            x_data = x_full[::data_subsample]
            y_data = y_full[::data_subsample]
            
            del x_full, y_full
            force_cleanup()
            
            # Compute coordinates for subsampled data
            X_data, Y_data = np.meshgrid(x_data, y_data)
            X_data = X_data.astype(np.float32)
            Y_data = Y_data.astype(np.float32)
            
            # Quick coordinate computation
            a_var = np.sin(X_data) ** 2 + (np.cos(X_data) ** 2 * (np.cos(Y_data) ** 2 + ((r_eq ** 2 / r_pol ** 2) * np.sin(Y_data) ** 2)))
            b_var = -2 * H * np.cos(X_data) * np.cos(Y_data)
            c_var = H ** 2 - r_eq ** 2
            
            discriminant = np.maximum(b_var ** 2 - 4 * a_var * c_var, 0)
            r_s = (-b_var - np.sqrt(discriminant)) / (2 * a_var)
            s_x = r_s * np.cos(X_data) * np.cos(Y_data)
            s_y = -r_s * np.sin(X_data)
            s_z = r_s * np.cos(X_data) * np.sin(Y_data)
            
            denominator = np.sqrt((H - s_x) ** 2 + s_y ** 2)
            denominator = np.where(denominator == 0, 1e-10, denominator)
            
            lats = np.rad2deg(np.arctan((r_eq ** 2 / r_pol ** 2) * (s_z / denominator))).astype(np.float32)
            lons = np.rad2deg(lambda_0 - np.arctan(s_y / (H - s_x))).astype(np.float32)
            
            # Clean coordinate computation arrays
            del X_data, Y_data, a_var, b_var, c_var, discriminant, r_s, s_x, s_y, s_z, denominator
            force_cleanup()
            
            progress_bar.progress(90, "Converting temperature...")
            
            # Get calibration and convert to temperature
            planck_fk1 = float(ds.variables['planck_fk1'][0])
            planck_fk2 = float(ds.variables['planck_fk2'][0])
            planck_bc1 = float(ds.variables['planck_bc1'][0])
            planck_bc2 = float(ds.variables['planck_bc2'][0])
            
            # Convert to temperature
            data = np.maximum(data, 1e-10)  # Avoid log(0)
            temp_data = (planck_fk2 / (np.log((planck_fk1 / data) + 1)) - planck_bc1) / planck_bc2 - 273.15
            temp_data = temp_data.astype(np.float32)
            
            del data
            force_cleanup()
            
            # Extract region
            region_mask = ((lons >= lon_min) & (lons <= lon_max) & 
                          (lats >= lat_min) & (lats <= lat_max))
            
            temp_region = np.ma.masked_where(~region_mask, temp_data)
            lons_region = np.ma.masked_where(~region_mask, lons)
            lats_region = np.ma.masked_where(~region_mask, lats)
            
            del temp_data, lons, lats, region_mask
            force_cleanup()
            
            progress_bar.progress(95, "Creating plot...")
            
            # Minimal plot
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100, 
                                  subplot_kw={'projection': ccrs.PlateCarree()})
            
            # Plot with minimal features
            custom_cmap = rbtop3()
            im = ax.pcolormesh(lons_region, lats_region, temp_region, 
                              cmap=custom_cmap, vmin=-100, vmax=40,
                              transform=ccrs.PlateCarree(), shading='nearest')
            
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            
            # Minimal map features
            ax.coastlines(resolution='110m')  # Lowest resolution
            ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
            
            # Simple colorbar
            plt.colorbar(im, ax=ax, orientation='vertical', label='Temperature (°C)', shrink=0.7)
            
            # Title
            date_time_str = filename.split('_')[3][1:12]
            dt = datetime.strptime(date_time_str, '%Y%j%H%M')
            sat_name = 'GOES-16' if 'G16' in filename else 'GOES-17' if 'G17' in filename else 'GOES-18' if 'G18' in filename else 'GOES-19'
            title = dt.strftime(f'{sat_name} IR - %b %d, %Y %H:%M UTC')
            plt.title(title, fontsize=14, weight='bold')
            
            fig.text(0.5, 0.02, 'Sekai Chandra (@Sekai_WX)', ha='center', fontsize=10)
            
            # Final cleanup
            del temp_region, lons_region, lats_region
            force_cleanup()
            
            return fig
            
    except Exception as e:
        force_cleanup()
        raise

def process_goes_minimal(date_input, hour, center_coord):
    """Ultra-minimal processing pipeline"""
    
    if not processing_lock.acquire(blocking=False):
        raise Exception("Another request is processing. Please wait and try again.")
    
    try:
        year = date_input.year
        day_of_year = date_input.timetuple().tm_yday
        
        requested_date = datetime(year, date_input.month, date_input.day, hour)
        center_lon, center_lat = center_coord
        
        progress_bar = st.progress(0, "Starting satellite request...")
        
        # Select satellite
        satellite_name, error = select_best_satellite(center_lon, center_lat, requested_date)
        if error:
            raise Exception(error)
        
        progress_bar.progress(5, f"Using {satellite_name}...")
        
        # Download
        file_path, filename = download_minimal(satellite_name, year, day_of_year, hour, progress_bar)
        
        try:
            # Process
            fig = process_minimal_region(file_path, filename, center_coord, progress_bar)
            progress_bar.progress(100, "Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            
            return fig, satellite_name
            
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            force_cleanup()
            
    finally:
        processing_lock.release()

# Streamlit UI
st.title("🛰️ GOES Satellite Viewer")
st.markdown("### Ultra-Lightweight High-Resolution Satellite Data")

st.warning("⚡ **Streamlit Optimized**: Heavy subsampling applied for reliable cloud processing (~2-4 minutes)")

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📅 Date & Time")
    
    today = datetime.now().date()
    min_date = datetime(2017, 4, 1).date()
    
    date_input = st.date_input("Date", value=today, min_value=min_date, max_value=today)
    hour_input = st.selectbox("Hour (UTC)", options=list(range(0, 24, 3)), index=4, format_func=lambda x: f"{x:02d}:00")
    
    st.subheader("🌍 Location")
    
    location_method = st.radio("Input Method", ["City Name", "Coordinates"])
    
    if location_method == "City Name":
        location_input = st.text_input("Location", placeholder="Miami, Denver, Phoenix")
        lat, lon = None, None
        if location_input:
            with st.spinner("Finding location..."):
                lat, lon = get_coordinates(location_input)
                if lat and lon:
                    st.success(f"📍 {lat:.2f}°, {lon:.2f}°")
                else:
                    st.error("Not found")
    else:
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitude", min_value=-60.0, max_value=60.0, value=25.8, step=1.0)
        with col_lon:
            lon = st.number_input("Longitude", min_value=-160.0, max_value=-40.0, value=-80.2, step=1.0)
    
    # Preview
    if lat is not None and lon is not None:
        try:
            req_date = datetime(date_input.year, date_input.month, date_input.day, hour_input)
            selected_sat, _ = select_best_satellite(lon, lat, req_date)
            if selected_sat:
                st.info(f"🛰️ **{selected_sat}**")
        except:
            pass
    
    generate_button = st.button("🚀 Generate (Allow 2-4 min)", type="primary")

with col2:
    st.subheader("📊 Satellite Image")
    
    if generate_button:
        if lat is not None and lon is not None:
            try:
                fig, satellite_used = process_goes_minimal(date_input, hour_input, (lon, lat))
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                force_cleanup()
                
                st.success(f"✅ {satellite_used} data processed!")
                
            except Exception as e:
                st.error(f"❌ {str(e)}")
                if "Another request" in str(e):
                    st.info("💡 Only one user can process at a time. Please wait and retry.")
                else:
                    st.info("💡 Try a different time/location if error persists.")
        else:
            st.warning("⚠️ Please provide a valid location")

with st.expander("⚡ Performance Info"):
    st.markdown("""
    **Ultra-Lightweight Mode Active:**
    - **8x data subsampling** for memory efficiency
    - **6° region size** (reduced from 10°)
    - **Single user processing** to prevent overload
    - **Progressive cleanup** throughout processing
    - **3-hour intervals** for better reliability
    
    **Trade-offs:**
    - Longer processing time (2-4 minutes)
    - Lower spatial resolution 
    - Smaller coverage area
    - Single concurrent user
    
    **Result:** Reliable processing on Streamlit free tier! 🎯
    """)

st.markdown("---")
st.markdown("*Ultra-optimized for Streamlit Cloud by Sekai Chandra (@Sekai_WX)*")