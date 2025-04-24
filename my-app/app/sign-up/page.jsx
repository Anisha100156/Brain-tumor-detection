import { SignUp } from '@clerk/nextjs';

export default function Page() {
  return  <div className="h-screen w-full flex items-center justify-center bg-gray-100">
  <SignUp path="/sign-up" routing="path"  />
</div>
}
