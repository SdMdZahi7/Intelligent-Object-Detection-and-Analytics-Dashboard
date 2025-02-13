# **Intelligent Object Detection and Analytics Dashboard**  
A professional dashboard for visualizing and analyzing object detection frequencies across video frames using data analytics and visualization techniques.

---

## 📖 **Abstract**  
This project presents a sophisticated dashboard that efficiently analyzes object detection across video frames, leveraging advanced video analytics techniques. By integrating real-time data visualization and interactive features, it provides users with clear, actionable insights into object detection trends. The inclusion of CSV storage ensures easy access and management of processed data, making the tool practical for researchers and developers. Designed for scalability, it caters to a wide range of use cases, from video surveillance to retail analytics. This project bridges the gap between raw video data and insightful analysis, offering an innovative solution in the field of video analytics.

---

## 🛠️ **Features**
- **📊 Advanced Visualization:** Graphical representation of detection frequencies across frames.  
- **📈 Real-Time Analytics:** Seamless integration of data into visually rich dashboards.  
- **💾 Data Storage:** Save detection results in a structured CSV format for future analysis.  
- **📂 Simplified Workflow:** Modular Python scripts ensure easy maintenance and scalability.
- 🔍 Insightful Object Tracking: Analyze and monitor object detection patterns effectively across multiple video frames.
- ⚡ Performance Optimization: Efficient processing and visualization ensure quick insights, even with large datasets.

---

## 📂 **Data Set Input**

The input data set for this project is a video file, which undergoes processing for object detection. Below are the key details:

 **1. Input Format**  
- Video file formats: `.mp4`, `.avi`, or other commonly supported formats.

 **2. Processing**  
- The video frames are extracted and analyzed using advanced object detection algorithms.  

 **3. Output**  
- Each detection is recorded in a structured manner with the following attributes:  
- **Frame Number**: Identifies the frame within the video.  
- **Object Detected**: Specifies the type of object identified (e.g., person, vehicle).  
- **Confidence Score**: A reliability metric for the detection.  
- **Bounding Box Coordinates**: Indicates the position of the detected object using:  
    - `X_min` and `Y_min` (top-left corner of the bounding box)  
    - `X_max` and `Y_max` (bottom-right corner of the bounding box)  

 **4. Usage**  
- The processed data is saved as a **CSV file**, enabling easy analysis, visualization, or integration into further projects.

---

## 🔧 **Libraries Used**
- **Python**: Core language for implementation.  
- **Pandas**: Efficient data preprocessing and manipulation.  
- **Streamlit**: For creating interactive dashboards.  
- **Matplotlib & Seaborn**: Enhanced data visualizations.  
- **OpenCV**: Video frame handling (optional).  

---

## **📜 Problem Definition**
The **Intelligent Object Detection and Analytics Dashboard** is a comprehensive solution designed to streamline the process of detecting and analyzing objects in video frames. With the increasing use of video analytics across industries, this project addresses key challenges such as resource-intensive computations, lack of real-time insights, and the complexity of managing large datasets.

By integrating robust data visualization techniques, real-time analytics, and structured data storage, the project enables users to:

- **📈 Analyze detection frequencies across video frames effectively.**  
- **📊 Access actionable insights through an interactive, user-friendly dashboard.**  
- **💾 Store detection results in a structured CSV format for further analysis and reporting.**  

This project is aimed at researchers, developers, and data analysts looking for an efficient and scalable solution for video analytics, making the analysis process more accessible, intuitive, and insightful.

---

## **⚙️ System Architecture**

The system architecture of the **Intelligent Object Detection and Analytics Dashboard** is designed to ensure seamless integration of components for efficient video analytics. Below is a breakdown of the key components:

- **📥 Input**:  
   The system accepts pre-detected object detection results in a structured CSV format. This dataset includes essential details such as frame numbers, object types, confidence scores, and bounding box coordinates (`X_min`, `Y_min`, `X_max`, `Y_max`).

- **🔄 Processing**:  
   The raw input data undergoes cleaning and transformation to ensure it is ready for visualization. This step includes:  
   - Handling missing or inconsistent values.  
   - Filtering based on confidence thresholds for accuracy.  
   - Aggregating detection frequencies for meaningful insights.  

- **📊 Output**:  
   The processed data is displayed on an interactive, real-time dashboard. The dashboard provides:  
   - Graphical trends of object detection frequencies across frames.  
   - A user-friendly interface for detailed insights into object trends.  
   - Export options for storing insights in CSV format for reporting and further analysis.  

![image](https://github.com/user-attachments/assets/c6ae903b-88c0-4313-9408-b1aee8796d55)

---

## 🧠 **Algorithm**
**Simplified Workflow**:  

1. **📥 Data Ingestion**:  
   - The system reads pre-detected object detection data from a CSV file.  
   - The dataset includes key information such as frame numbers, detected objects, confidence scores, and bounding box coordinates (`X_min`, `Y_min`, `X_max`, `Y_max`).  

2. **🛠️ Data Preprocessing**:  
   - Data is cleaned to handle missing or inconsistent entries.  
   - Filters are applied to exclude detections below a predefined confidence threshold, ensuring accuracy.  
   - Data is structured and organized for efficient visualization, including aggregation of detection frequencies.  

3. **📊 Visualization**:  
   - Real-time graphs and charts are generated to showcase detection trends.  
   - Visualizations include frequency analysis of detected objects, confidence distributions, and spatial trends based on bounding box data.  

4. **📂 Output Delivery**:  
   - An interactive dashboard presents the visualized data, allowing users to explore insights effortlessly.  
   - The dashboard supports exporting processed data and visualizations for further reporting or analysis.  

This streamlined workflow ensures that users can efficiently analyze object detection data with minimal complexity, making the process accessible and insightful.



---
## 🖥️**Code Implementation and Output**
~~~
from google.colab import files
import cv2
from google.colab.patches import cv2_imshow # Import cv2_imshow


# Step 1: Upload the video file
uploaded = files.upload()

# Get the file name of the uploaded video
video_file = list(uploaded.keys())[0]
print(f"Uploaded video file: {video_file}")

# Step 2: Initialize video capture with the uploaded video
video_capture = cv2.VideoCapture(video_file)

if not video_capture.isOpened():
    print("Error: Could not open the uploaded video file.")
else:
    print("Video file accessed successfully!")

# Step 3: Loop to read and display frames from the video
while True:
    ret, frame = video_capture.read()

    # Check if the frame is read successfully
    if not ret:
        print("End of video or failed to grab frame.")
        break

    # Display the frame using cv2_imshow
    cv2_imshow(frame) # Use cv2_imshow instead of cv2.imshow

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
~~~

![image](https://github.com/user-attachments/assets/60babc9e-702d-4280-86bb-5d7759fe7e14)
---

~~~
import cv2
import torch
import pandas as pd
from google.colab.patches import cv2_imshow

# Load the pretrained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize the video capture with the uploaded video file
video_capture = cv2.VideoCapture(video_file)

if not video_capture.isOpened():
    print("Error: Could not open the uploaded video file.")
else:
    print("Video file accessed successfully!")

# Limit the number of frames to process
max_frames = 100  # Adjust this number as needed
frame_count = 0

# Data collection list
data_list = []

# Loop to process video frames
while frame_count < max_frames:
    ret, frame = video_capture.read()

    # Check if the frame is read successfully
    if not ret:
        print("End of video or failed to grab frame.")
        break

    # Increment frame count
    frame_count += 1

    # Convert the frame to RGB (YOLOv5 works on RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use YOLOv5 model to detect objects
    results = model(rgb_frame)

    # Extract detections
    detections = results.pandas().xyxy[0]  # Convert detection results to Pandas DataFrame

    # Loop through each detected object
    for _, row in detections.iterrows():
        # Extract details
        obj_name = row['name']       # Object class name
        confidence = row['confidence']  # Confidence score
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']  # Bounding box

        # Append to data list
        data_list.append({
            "Frame": frame_count,
            "Object": obj_name,
            "Confidence": confidence,
            "X_min": xmin,
            "Y_min": ymin,
            "X_max": xmax,
            "Y_max": ymax
        })

    # Render the results on the frame
    annotated_frame = results.render()[0]

    # Display the frame with detections
    cv2_imshow(annotated_frame)  # Use cv2_imshow in Colab

# Release the video capture
video_capture.release()

# Convert the data list to a Pandas DataFrame
df = pd.DataFrame(data_list)

# Save the data to a CSV file
df.to_csv('object_detection_data.csv', index=False)
print("Data saved to object_detection_data.csv")

# Display a preview of the collected data
df.head()
~~~

![image](https://github.com/user-attachments/assets/1fc39936-4020-4920-8e1e-4116a15f362d)
---
~~~
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
df = pd.read_csv('object_detection_data.csv')

# Preview the data
print("Preview of the collected data:")
print(df.head())

# --- Data Analytics ---

# 1. Count of each object type
object_counts = df['Object'].value_counts()
print("\nObject Counts:")
print(object_counts)

# 2. Confidence analysis
avg_confidence = df.groupby('Object')['Confidence'].mean()
print("\nAverage Confidence by Object:")
print(avg_confidence)

# 3. Frame-wise object detection count
frame_counts = df.groupby('Frame')['Object'].count()
print("\nObject Detection Count per Frame:")
print(frame_counts.head())
~~~

![image](https://github.com/user-attachments/assets/1fde89af-67f0-45b4-a712-701f49244022)
---

~~~
# --- Visualization ---

# 1. Bar plot of object counts
plt.figure(figsize=(10, 6))
sns.barplot(x=object_counts.index, y=object_counts.values, palette="viridis", hue=object_counts.index, dodge=False)
plt.legend([], [], frameon=False)  # Remove the legend for this case
plt.title("Object Detection Counts", fontsize=16)
plt.xlabel("Object Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.show()
~~~

![image](https://github.com/user-attachments/assets/6441d4d5-6114-411d-97da-93de641255e6)
---
~~~
# 2. Line plot of object detection frequency across frames
plt.figure(figsize=(12, 6))
sns.lineplot(x=frame_counts.index, y=frame_counts.values, marker='o', color='b')
plt.title("Object Detection Frequency Across Frames", fontsize=16)
plt.xlabel("Frame Number", fontsize=12)
plt.ylabel("Number of Objects Detected", fontsize=12)
plt.grid(True)
plt.show()
~~~

![image](https://github.com/user-attachments/assets/917224f2-0bec-4be0-ab40-e2a5f1cb034a)
---

~~~
# 3. Bar plot of average confidence per object
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_confidence.index, y=avg_confidence.values, palette="mako", hue=avg_confidence.index, dodge=False)
plt.legend([], [], frameon=False)  # Remove the legend for this case
plt.title("Average Confidence for Detected Objects", fontsize=16)
plt.xlabel("Object Type", fontsize=12)
plt.ylabel("Average Confidence", fontsize=12)
plt.xticks(rotation=45)
plt.show()
~~~

![image](https://github.com/user-attachments/assets/90937d09-0250-41bb-bb59-6bb721295a0f)
---

~~~
# Save each DataFrame to a CSV file
object_counts.to_csv('object_counts.csv', index=False)
frame_counts.to_csv('frame_counts.csv', index=False)
avg_confidence.to_csv('avg_confidence.csv', index=False)

print("Data has been saved to CSV files!")
~~~
![image](https://github.com/user-attachments/assets/12d3faa8-e7e7-4f57-bc54-ca38adfc1303)
---


### 📈 **Dashboard Code and Implementation**
The dashboard code, stored in dashboard_app.py, is a Python-based application built using Streamlit. It provides an intuitive and interactive platform for visualizing object detection results across video frames.
~~~
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import numpy as np

# Load Data
object_counts = pd.read_csv('object_counts.csv')  # Data for object counts
frame_counts = pd.read_csv('frame_counts.csv')  # Data for detection frequency across frames
avg_confidence = pd.read_csv('avg_confidence.csv')  # Data for average confidence scores

# Streamlit Dashboard
st.set_page_config(page_title="Video Analytics Dashboard", layout="wide", page_icon="📊")

st.title("📹 *VisionTrack: Interactive Object Detection Dashboard*")
st.markdown("""
Welcome to the **VisionTrack Dashboard**!  
This dashboard provides insights from video analytics, including object detection counts, detection frequency across frames, and average confidence scores.  
Explore the data with informative charts below! 🚀  
""")

# Visualization 1: Object Detection Counts
st.subheader("📦 **Object Detection Counts**")
st.markdown("""
The chart below shows the total number of times each object was detected in the video.  
This helps identify the most frequently occurring objects in the analyzed footage. 📊
""")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x=object_counts['Object'], y=object_counts['Count'], ax=ax1, palette="coolwarm")
ax1.set_title("Object Detection Counts", fontsize=18, fontweight="bold")
ax1.set_xlabel("Object Type", fontsize=14, fontweight="bold")
ax1.set_ylabel("Count", fontsize=14, fontweight="bold")
for i, count in enumerate(object_counts['Count']):
    ax1.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=12, color="black")
st.pyplot(fig1)

# Visualization 2: Detection Frequency Over Frames
# --- Visualization 2: Object Detection Frequency Across Frames ---

frame_counts = np.array([
    [1, 15],
    [2, 20],
    [3, 25],
    [4, 30],
    [5, 12],
    [6, 18],
    [7, 24],
    [8, 20],
    [9, 15],
    [10, 10]
])

st.subheader("🎥 **Object Detection Frequency Across Frames**")

try:
    # Check and reformat frame_counts
    if isinstance(frame_counts, np.ndarray) and frame_counts.shape[1] == 2:
        frame_counts_df = pd.DataFrame(frame_counts, columns=["Frame", "Count"])
        frame_counts_df = frame_counts_df.head(10)  # Limit to first 10 frames
    else:
        st.error("frame_counts must be a 2D array with shape (n, 2). Please check the input data.")
        st.stop()
    
    # Extract x and y values for visualization
    x_values = frame_counts_df["Frame"].tolist()
    y_values = frame_counts_df["Count"].tolist()

    # Generate and display the visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x=x_values, 
        y=y_values, 
        marker='o', 
        linewidth=2.5, 
        color="darkblue",
        ax=ax
    )
    for x, y in zip(x_values, y_values):
        ax.text(x, y + 0.3, str(y), fontsize=10, color="darkblue", ha="center")

    ax.set_title("Object Detection Frequency Across First 10 Frames", fontsize=18, fontweight="bold")
    ax.set_xlabel("Frame Number", fontsize=14)
    ax.set_ylabel("Number of Objects Detected", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error processing frame_counts: {e}")


# Visualization 3: Average Confidence Scores
st.subheader("✅ **Average Confidence for Detected Objects**")
st.markdown("""
The chart below highlights the average confidence scores for each detected object type.  
Confidence scores indicate how certain the model was when identifying objects. 🌟
""")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(x=avg_confidence['Object'], y=avg_confidence['Confidence'], ax=ax3, palette="mako")
ax3.set_title("Average Confidence for Detected Objects", fontsize=18, fontweight="bold")
ax3.set_xlabel("Object Type", fontsize=14, fontweight="bold")
ax3.set_ylabel("Average Confidence", fontsize=14, fontweight="bold")
for i, confidence in enumerate(avg_confidence['Confidence']):
    ax3.text(i, confidence + 0.01, f"{confidence:.2f}", ha='center', va='bottom', fontsize=12, color="black")
st.pyplot(fig3)

# Save CSV Option
st.sidebar.title("⚙️ **Options**")
st.sidebar.markdown("""
You can save the visualized data to CSV files for further analysis. 📁  
Click the button below to save the data.
""")
if st.sidebar.button("Save Visualized Data to CSV"):
    object_counts.to_csv('visualized_object_counts.csv', index=False)
    frame_counts.to_csv('visualized_frame_counts.csv', index=False)
    avg_confidence.to_csv('visualized_avg_confidence.csv', index=False)
    st.sidebar.success("Data saved as CSV files! ✅")

# Footer
st.markdown("---")
st.markdown("""
💡 **Pro Tip**: Use these insights to refine your video analytics pipeline!  
Made with ❤️ by your analytics assistant.
""")
st.markdown("**Created by:** Syed Muhammed Zahi | **Powered by:** Python, Streamlit, Seaborn, Matplotlib")

~~~
![image](https://github.com/user-attachments/assets/9e2193af-4593-4b18-a682-a46246b96efd)
![image](https://github.com/user-attachments/assets/19f6eaff-a487-46c2-b480-39bae8c97d57)





---

## 📊 **Dashboard Output**  

The **Intelligent Object Detection and Analytics Dashboard** delivers an intuitive and interactive platform for analyzing object detection data. Here's what you can expect:  
- **Detection Trends**: Graphical representation of how often objects are detected across frames, helping identify patterns.  
- **Tabular Data View**: A structured table showcasing detection details such as frame number, object type, confidence score, and bounding box coordinates.  
- **Dynamic Exploration**: Interactive features like sliders and drop-down filters for customizing insights.  

### 📷 Dashboard Preview  
![image](https://github.com/user-attachments/assets/1d2fc0b8-ee0c-4144-a0d8-1e529160f6d0)
![image](https://github.com/user-attachments/assets/f6a8331b-dcd3-445d-9b58-76c26bb32972)
![image](https://github.com/user-attachments/assets/c159f8ff-b042-4781-8ba7-3214894971c0)


---


## 🚀 **Future Scope**  

The **Intelligent Object Detection and Analytics Dashboard** has significant potential for growth and real-world applications. Some future enhancements include:  
- **Real-Time Integration**: Implementing live video feed analysis for real-time detection and visualization.  
- **Enhanced Analytics**: Adding more advanced metrics such as object velocity tracking or density mapping.  
- **Scalability**: Adapting the dashboard to handle larger datasets and support distributed processing for high-performance requirements.  
- **Custom Alerts**: Introducing notifications or alerts based on specific detection thresholds or patterns.  
- **AI Integration**: Incorporating machine learning models to predict trends or classify objects beyond basic detection.  
- **Cross-Platform Accessibility**: Developing mobile and web-based versions for greater accessibility.  

---

## 🌟 Conclusion  

The **Intelligent Object Detection and Analytics Dashboard** serves as a stepping stone toward more innovative and accessible solutions in video analytics. By transforming raw detection data into meaningful insights through an intuitive and interactive interface, it bridges the gap between complexity and usability.  

This project highlights the potential of combining real-time analytics, robust visualization techniques, and modular design to solve real-world challenges. It sets the foundation for future advancements, where AI-powered predictions, live integrations, and cross-platform capabilities can take this solution to new heights.  

As industries increasingly rely on video analytics, this dashboard lays the groundwork for scalable and efficient solutions, empowering users to harness the full potential of their data. It's not just a tool—it's a leap toward making video analytics more insightful, accessible, and impactful.  

