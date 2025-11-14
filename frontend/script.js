const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const uploadBtn = document.getElementById("uploadBtn");
const resultBox = document.getElementById("resultBox");
const resultText = document.getElementById("resultText");
const previewBox = document.getElementById("previewBox");
const previewImage = document.getElementById("previewImage");

// When user selects a file show filename + preview
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];

  if (file) {
    fileName.textContent = file.name;

    // use object URL for preview
    const url = URL.createObjectURL(file);
    previewImage.src = url;
    previewBox.style.display = "block";
    previewBox.setAttribute("aria-hidden", "false");
  } else {
    fileName.textContent = "No file selected";
    previewBox.style.display = "none";
    previewBox.setAttribute("aria-hidden", "true");
  }

  // hide previous result if any
  resultBox.style.display = "none";
});

// Upload and call backend
uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select an image first!");
    return;
  }

  // show a quick "processing" message while waiting
  resultBox.style.display = "block";
  resultText.textContent = "Uploading image and waiting for prediction...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      // server returned an error
      const err = await res.json().catch(() => ({}));
      resultText.textContent = err?.error || `Server error (${res.status})`;
      return;
    }

    const data = await res.json();

    // Safety: if backend returns confidence as NaN or missing, handle gracefully
    const conf = typeof data.confidence === "number" ? data.confidence : 0;
    const confPct = (conf * 100).toFixed(2);

    // Show the prediction plus confidence
    resultText.textContent = `Prediction: ${data.class_name} (Confidence: ${confPct}%)`;
  } catch (err) {
    resultText.textContent = "Error connecting to backend.";
    console.error("Upload error:", err);
  }
});