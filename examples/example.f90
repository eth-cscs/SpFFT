program main
    use iso_c_binding
    use spfft
    implicit none
    integer :: i, j, k, counter
    integer, parameter :: dimX = 2
    integer, parameter :: dimY = 2
    integer, parameter :: dimZ = 2
    integer, parameter :: maxNumLocalZColumns = dimX * dimY
    integer, parameter :: processingUnit = 1
    integer, parameter :: maxNumThreads = -1
    type(c_ptr) :: grid = c_null_ptr
    type(c_ptr) :: transform = c_null_ptr
    integer :: errorCode = 0
    integer, dimension(dimX * dimY * dimZ * 3):: indices = 0
    complex(C_DOUBLE_COMPLEX), dimension(dimX * dimY * dimZ):: frequencyElements
    real(C_DOUBLE), dimension(2*dimX * dimY * dimZ):: spaceDomain
    complex(C_DOUBLE_COMPLEX), pointer :: spaceDomainPtr(:,:,:)
    type(c_ptr) :: realValuesPtr


    counter = 0
    do k = 1, dimZ
        do j = 1, dimY
            do i = 1, dimX
             frequencyElements(counter + 1) = cmplx(counter, -counter)
             indices(counter * 3 + 1) = i - 1
             indices(counter * 3 + 2) = j - 1
             indices(counter * 3 + 3) = k - 1
             counter = counter + 1
            end do
        end do
    end do

    ! print input
    print *, "Input:"
    do i = 1, size(frequencyElements)
         print *, frequencyElements(i)
    end do


    ! create grid
    errorCode = spfft_grid_create(grid, dimX, dimY, dimZ, maxNumLocalZColumns, processingUnit, maxNumThreads);
    if (errorCode /= SPFFT_SUCCESS) error stop

    ! create transform
    ! Note: A transform handle can be created without a grid if no resource sharing is desired.
    errorCode = spfft_transform_create(transform, grid, processingUnit, 0, dimX, dimY, dimZ, dimZ,&
        size(frequencyElements), SPFFT_INDEX_TRIPLETS, indices)
    if (errorCode /= SPFFT_SUCCESS) error stop

    ! grid can be safely destroyed after creating all required transforms
    errorCode = spfft_grid_destroy(grid)
    if (errorCode /= SPFFT_SUCCESS) error stop


    ! *************************************************
    ! Option A: Reuse internal buffer for space domain
    ! *************************************************

    ! set space domain array to use memory allocted by the library
    errorCode = spfft_transform_get_space_domain(transform, processingUnit, realValuesPtr)
    if (errorCode /= SPFFT_SUCCESS) error stop

    ! transform backward
    errorCode = spfft_transform_backward(transform, frequencyElements, processingUnit)
    if (errorCode /= SPFFT_SUCCESS) error stop


    call c_f_pointer(realValuesPtr, spaceDomainPtr, [dimX,dimY,dimZ])

    print *, ""
    print *, "After backward transform:"
    do k = 1, size(spaceDomainPtr, 3)
        do j = 1, size(spaceDomainPtr, 2)
            do i = 1, size(spaceDomainPtr, 1)
             print *, spaceDomainPtr(i, j, k)
            end do
        end do
    end do


    ! **********************************************
    ! Option B: Use external buffer for space domain
    ! **********************************************

    ! transform backward
    errorCode = spfft_transform_backward_ptr(transform, frequencyElements, spaceDomain)
    if (errorCode /= SPFFT_SUCCESS) error stop

    ! transform forward
    errorCode = spfft_transform_forward_ptr(transform, spaceDomain, frequencyElements, SPFFT_NO_SCALING)
    if (errorCode /= SPFFT_SUCCESS) error stop

    print *, ""
    print *, "After forward transform (without normalization):"
    do i = 1, size(frequencyElements)
             print *, frequencyElements(i)
    end do

    ! destroying the final transform will free the associated memory
    errorCode = spfft_transform_destroy(transform)
    if (errorCode /= SPFFT_SUCCESS) error stop

end
