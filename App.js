import React, { useState } from "react";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPrediction("");
    setError("");
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      alert("Please select a file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.prediction) {
        setPrediction(data.prediction); // Use the class name string directly
      } else {
        setError(data.error || "Unexpected error occurred.");
      }
    } catch (err) {
      setError("Fetch error: " + err.message);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>🌿 Crop Disease Detector</h1>
      <input type="file" onChange={handleFileChange} />
      <br />
      <button onClick={handlePredict} style={{ marginTop: "10px" }}>
        Predict
      </button>

      {prediction && <h2 style={{ color: "green" }}>Prediction: {prediction}</h2>}
      {error && <h2 style={{ color: "red" }}>Error: {error}</h2>}
    </div>
  );
}

export default App;
