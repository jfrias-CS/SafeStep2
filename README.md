# 🚶‍♀️ SafeStep: AI-Powered Sidewalk Damage Detection and Navigation
SafeStep is an Android application that uses TensorFlow Lite models to enhance pedestrian safety by detecting sidewalk hazards in real-time.

The application captures an image of the user's surroundings and employs detection and regression models to identify and quantify damages (e.g., cracks, uneven slabs). This raw data, including bounding boxes and calculated severity levels, is then sent via a structured prompt to a fine-tuned GPT-3.5 Turbo model.

The large language model (LLM) processes this technical data to generate natural language instructions. These instructions provide the user with clear, actionable advice on the precise location and severity of the damage, along with the next safest step or direction to take to avoid the hazard. SafeStep effectively translates complex machine learning outputs into intuitive, real-world guidance.
