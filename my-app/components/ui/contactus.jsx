import React from "react";

const Footer = () => {
  return (
    <footer className="bg-[#373f5b] text-white py-10 px-5">
     
      <div className="max-w-screen-xl mx-auto">
        <div className="md:flex justify-between items-start">
       
          <div className="mb-6 md:mb-0">
          <br></br><br></br>
            <p>RZ Block Nagar</p>
            <p>New Delhi, 110078</p>
            <p className="mt-2">📞 9832107954</p>
            <p>✉️ anishachowdhury34@gmail.com</p>
            <div className="flex space-x-4 mt-4">
              <a href="#">🌐</a>
              <a href="#">📘</a>
              <a href="#">🐦</a>
              <a href="#">🔗</a>
              <a href="#">▶️</a>
            </div>
          </div>

          {/* Center Column */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-sm">
            <div>
            <br></br><br></br>
              <h4 className="font-semibold mb-2">
                Signs & Symptoms of a Brain Tumor
              </h4>
              
              <ul className="space-y-1">
                <li>Brain Tumor FAQs</li>
              </ul>
              <h4 className="font-semibold mt-4 mb-2">About</h4>
              <ul className="space-y-1">
                <li>Team</li>
                <li>Board of Directors</li>
                <li>News</li>
                <li>Financials</li>
              </ul>
            </div>
            <div>
            <br></br><br></br>
              <h4 className="font-semibold mb-2">Get Support</h4>
              <ul className="space-y-1">
                <li>Is a Clinical Trial Right for Me?</li>
                <li>Patient Resources</li>
                <li>Caregiver Resources</li>
                <li>Education</li>
              </ul>
              <h4 className="font-semibold mt-4 mb-2">Contact Us</h4>
            </div>
            <div>
            <br></br><br></br>
              <h4 className="font-semibold mb-2">Navigation Services</h4>
              <ul className="space-y-1">
                <li>Our Impact</li>
                <li>Patient Experience</li>
              </ul>
              <div className="flex space-x-2 mt-2">
              
              </div>
              <h4 className="font-semibold mt-4 mb-2">Donate</h4>
            </div>
          </div>
        </div>

        <div className="border-t border-gray-400 mt-8 pt-4 flex flex-col md:flex-row justify-between items-center text-sm">
        
          <div className="flex items-center space-x-2">
           
            <a href="#" className="hover:underline">
              Terms & Conditions
            </a>
          </div>
          <p className="mt-2 md:mt-0">Web design by Anisha and Megha</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
