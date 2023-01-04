#include <Kokkos_Core.hpp>

void print_kokkos_configuration()
{
    std::cout << "Kokkos configuration: " << std::endl;
#ifdef KOKKOS_ENABLE_SERIAL
    std::cout << "  Kokkos::Serial is available." << std::endl;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    std::cout << "  Kokkos:OpenMP is available" << std::endl;
#endif
#ifdef KOKKOS_ENABLE_CUDA
    std::cout << "  Kokkos:CUDA is available" << std::endl;
#endif
#ifdef KOKKOS_ENABLE_HIP
    std::cout << "  Kokkos:HIP is available" << std::endl;
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    std::cout << "  Kokkos::OpenMPTarget is available" << std::endl;
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    std::stringstream ss;
    Kokkos::DefaultExecutionSpace{}.print_configuration(ss);
    const std::string& sinfo = ss.str();
    if (!sinfo.empty()) {
        std::cout << "Kokkos default execution space: " << std::endl;
        std::cout << sinfo << std::endl;
    } else {
        std::cout << "Assuming default space = host space" << std::endl;
    }
    std::cout << std::endl;
#else
    std::cout << "Default execution space = "
              << Kokkos::DefaultExecutionSpace::name()
              << std::endl << std::endl;
#endif
}

#ifdef KOKKOS_ENABLE_CUDA
using MemSpace = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using MemSpace = Kokkos::Experimental::HIPSpace;
#elif defined(KOKKOS_ENABLE_OPENMP)
using MemSpace = Kokkos::OpenMP;
#else
using MemSpace = Kokkos::HostSpace;
#endif

typedef Kokkos::View<unsigned*, MemSpace> UnsignedViewType;

class FakeFieldBase
{
public:
  KOKKOS_DEFAULTED_FUNCTION FakeFieldBase() = default;
  KOKKOS_DEFAULTED_FUNCTION FakeFieldBase(const FakeFieldBase&) = default;
  KOKKOS_DEFAULTED_FUNCTION FakeFieldBase(FakeFieldBase&&) = default;
  KOKKOS_FUNCTION FakeFieldBase& operator=(const FakeFieldBase&) { return *this; }
  KOKKOS_FUNCTION FakeFieldBase& operator=(FakeFieldBase&&) { return *this; }
  KOKKOS_FUNCTION virtual ~FakeFieldBase() {}
};

template<typename T>
struct FakeField : public FakeFieldBase
{
public:
  KOKKOS_FUNCTION FakeField()
  {
    printf("FakeField default ctor\n");
  }

  KOKKOS_FUNCTION FakeField(const FakeField<T>& src)
  {
    printf("FakeField copy ctor\n");
  }

  KOKKOS_FUNCTION ~FakeField()
  {
    printf("FakeField dtor\n");
  }

  KOKKOS_FUNCTION double get_val() const { return val; }

private:
  UnsignedViewType unsignedDeviceView1;
  typename UnsignedViewType::HostMirror unsignedHostView1;
  UnsignedViewType unsignedDeviceView2;
  typename UnsignedViewType::HostMirror unsignedHostView2;
  UnsignedViewType unsignedDeviceView3;
  typename UnsignedViewType::HostMirror unsignedHostView3;
  UnsignedViewType unsignedDeviceView4;
  typename UnsignedViewType::HostMirror unsignedHostView4;
  UnsignedViewType unsignedDeviceView5;
  typename UnsignedViewType::HostMirror unsignedHostView5;
  UnsignedViewType unsignedDeviceView6;
  typename UnsignedViewType::HostMirror unsignedHostView6;
  UnsignedViewType unsignedDeviceView7;
  typename UnsignedViewType::HostMirror unsignedHostView7;
  UnsignedViewType unsignedDeviceView8;
  typename UnsignedViewType::HostMirror unsignedHostView8;
  UnsignedViewType unsignedDeviceView9;
  typename UnsignedViewType::HostMirror unsignedHostView9;
  UnsignedViewType unsignedDeviceView10;
  typename UnsignedViewType::HostMirror unsignedHostView10;
  T val = 0.0;
};

struct MyFakeDeviceClass
{
  KOKKOS_DEFAULTED_FUNCTION MyFakeDeviceClass() = default;
  KOKKOS_FUNCTION MyFakeDeviceClass(const MyFakeDeviceClass& src)
  : field(src.field), num(src.num)
  {}
  KOKKOS_DEFAULTED_FUNCTION ~MyFakeDeviceClass() = default;

  KOKKOS_FUNCTION unsigned get_num() const { return num; }

  FakeField<double> field;
  unsigned num = 0;
};

void test_fake_field_placement_new()
{
  MyFakeDeviceClass hostObj;
  hostObj.num = 42;

  printf("sizeof(MyFakeDeviceClass): %lu, sizeof(FakeField): %lu\n", sizeof(MyFakeDeviceClass), sizeof(FakeField<double>));
  std::string debugName("MyFakeDeviceClass");
  MyFakeDeviceClass* devicePtr = static_cast<MyFakeDeviceClass*>(Kokkos::kokkos_malloc<MemSpace>(debugName, sizeof(MyFakeDeviceClass)));

  int constructionFinished = 0;
  printf("about to call parallel_reduce for placement new\n");
  Kokkos::parallel_reduce(1, KOKKOS_LAMBDA(const unsigned& i, int& localFinished) {
    printf("before placement-new\n");
    new (devicePtr) MyFakeDeviceClass(hostObj);
    printf("after placement-new\n");
    localFinished = 1;
  }, constructionFinished);

  int numFromDevice = 0;
  printf("about to call parallel_reduce for access check\n");
  Kokkos::parallel_reduce(1, KOKKOS_LAMBDA(const unsigned& i, int& localNum) {
    localNum = devicePtr->get_num();
  }, numFromDevice);

  Kokkos::kokkos_free<MemSpace>(devicePtr);

  if (constructionFinished==1)
	  printf("\nConstruction Test Passed!\n");
  else
	  printf("\nConstruction Test Failed!\n");

  if (numFromDevice==42)
	  printf("Test Passed!\n");
  else
	  printf("Test Failed!\n");
}


int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        print_kokkos_configuration();
		  test_fake_field_placement_new();
    }
    Kokkos::finalize();
    return 0;
}

