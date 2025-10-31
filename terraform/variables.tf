variable "project_id" {
  description = "GCP project where the GPU instance will be created."
  type        = string
  default     = "depth-aware"
}

variable "region" {
  description = "Region that hosts the subnet and static IP."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone where the compute instance will run."
  type        = string
  default     = "us-east4-c"
}

variable "machine_type" {
  description = "Compute Engine machine type that provides at least 16GB memory."
  type        = string
  default     = "n1-highmem-4"
}

variable "gpu_type" {
  description = "GPU accelerator type to attach."
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "perma_disk_size_gb" {
  description = "Size of the additional persistent SSD disk."
  type        = number
  default     = 100
}
