import Image from "next/image";

export default function SocialWorkServices() {
  return (
    <div className="bg-[#76B5A2] py-20 px-4 md:px-10 text-white">
      <div className="relative grid grid-cols-[1fr_auto_1fr] gap-x-6 gap-y-16 items-start">
        {/* Left column */}
        <div className="space-y-30">
          <div className="flex items-start gap-4">
            <Image src="/newsad.png" alt="insurance" width={100} height={100} />
            <div>
              <h2 className="font-bold text-lg">Meningioma</h2>
              <p className="text-lg">
                Meningiomas are typically slow-growing tumors that form in the membranes surrounding the brain and spinal cord. While usually benign, they can still cause symptoms based on their size and location. Early detection is key for managing them effectively through surgery or observation.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-4">
            <Image src="/newsad.png" alt="emotional support" width={100} height={100} />
            <div>
              <h2 className="font-bold text-lg">Pituitary Tumor</h2>
              <p className="text-lg">
                Pituitary tumors develop in the pituitary gland at the base of the brain, which controls vital hormones. These tumors can affect vision, growth, and hormone levels. Many are noncancerous and treatable through medication, surgery, or radiation.
              </p>
            </div>
          </div>
        </div>

        {/* Center heart icon */}
        <div className="flex justify-center items-center relative top-25">
          <div className="bg-[#5C9984] p-6 rounded-full">
            <Image src="/heartmed.png" alt="heart" width={80} height={80} />
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-30">
          <div className="flex items-start gap-4">
            <Image src="/newsad.png" alt="finance" width={100} height={100} />
            <div>
              <h2 className="font-bold text-lg">Glioma</h2>
              <p className="text-lg">
                Gliomas are a group of tumors arising from glial cells in the brain or spine. They vary in severity, with some types being aggressive. Treatment may include surgery, radiation, and chemotherapy. Early diagnosis can improve outcomes significantly.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-4">
            <Image src="/new.png" alt="planning" width={80} height={80} />
            <div>
              <h2 className="font-bold text-lg">No Tumor</h2>
              <p className="text-lg">
                If no tumor is detected, it means the MRI shows no signs of abnormal brain masses. This result brings relief but may prompt further investigation if symptoms persist. Always follow up with a medical professional for personalized guidance.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
