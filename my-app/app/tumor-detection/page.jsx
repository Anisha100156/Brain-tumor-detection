"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export default function TumorDetection() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select a file first.");
      return;
    }

    setError("");
    setLoading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setResult({
        message: data.result,
        confidence: data.confidence,
        imageUrl: data.image_url,
      });
    } catch (error) {
      setError(error.message || "An error occurred while analyzing the image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#B7D3F2] to-[#D8B4FE] p-10">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6 }}
        className="max-w-3xl w-full bg-white shadow-2xl rounded-3xl p-10 border border-gray-200"
      >
        <h1 className="text-4xl font-extrabold text-[#1E3A8A] text-center">
          MRI Tumor Detection System
        </h1>
        <p className="text-gray-600 text-center mt-2">
          Upload an MRI image to analyze for tumor detection with AI assistance.
        </p>

        <div className="mt-8 space-y-5">
          <label className="block text-[#1E3A8A] font-semibold">Select MRI Image:</label>
          <input
            type="file"
            onChange={handleFileChange}
            className="w-full border border-gray-300 rounded-xl p-3 text-gray-800 bg-white focus:outline-none focus:ring-2 focus:ring-violet-500 transition shadow"
          />
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleUpload}
            className="w-full bg-gradient-to-r from-indigo-500 to-violet-500 text-white py-3 rounded-xl hover:from-indigo-600 hover:to-violet-600 transition shadow-lg font-semibold text-lg"
          >
            {loading ? "Analyzing..." : "Upload and Analyze"}
          </motion.button>
        </div>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mt-4 text-center text-red-600 font-semibold"
          >
            {error}
          </motion.div>
        )}

        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mt-8 p-6 rounded-2xl bg-gradient-to-br from-green-100 to-green-200 border border-green-400 text-center shadow-xl"
          >
            <h2 className="text-2xl font-bold text-green-700">
              {result.message}
            </h2>
            <p className="text-gray-800 mt-2">Confidence: {result.confidence}</p>
            {result.imageUrl && (
              <img
                src={result.imageUrl}
                alt="Uploaded MRI"
                className="mt-4 w-full rounded-xl shadow-md border border-gray-300"
              />
            )}
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
