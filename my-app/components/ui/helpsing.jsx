import React from "react";
import { useRouter } from "next/navigation";
import { Play } from "lucide-react";
import { TextGenerateEffect } from "./text-generate";
import { TextGenerate } from "./text-gen";
const help="About"
const HelpSections = () => {
  const router = useRouter();
  return (
    <section className="relative py-16 px-6 bg-white">
  
      <div className="absolute left-0 bottom-0 opacity-5 z-0">
       
        <img src="	https://149587026.v2.pressablecdn.com/wp-content/t…heme/assets/images/brain-graphic-background-c.svg" alt="Brain Icon" className="w-64 h-64" />
      </div>

      <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center relative z-10 gap-10">
        
        <div className="flex-1 text-left">
         <TextGenerate words={help}/>
          <p className="text-[#212636] mb-4 leading-relaxed font-serif text-lg">
          A brain tumor is an abnormal growth of cells within the brain or its surrounding structures. These tumors can be benign (non-cancerous) or malignant (cancerous), and they vary widely in their growth rates and effects on brain function. Brain tumors may originate in the brain itself (primary tumors) or spread from other parts of the body (secondary or metastatic tumors). The impact of a brain tumor depends on its size, type, and location, as even a benign tumor can be life-threatening if it compresses vital brain structures. Diagnosis and treatment typically involve imaging studies such as MRI or CT scans, followed by biopsy and a combination of surgery, radiation, or chemotherapy based on the type of tumor.
          </p>
          <ul className="list-disc list-inside text-[#212636] mb-6 font-serif text-lg leading-relaxed">
  
</ul>

          
        </div>

        {/* Right Content - Image with Play Button */}
        <div className="flex-1 relative">
          <img
            src="/tumor.jpeg"
            alt="Couple on bench"
            className="rounded-lg w-full h-130 object-cover"
          />
          <button className="absolute inset-0 flex justify-center items-center">
            
          </button>
        </div>
      </div>
    </section>
  );
};

export default HelpSections;
