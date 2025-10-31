terraform {
  required_version = ">= 1.3.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

locals {
  perma_disk_name = "perma-disk"
}

resource "google_compute_instance" "vscode_gpu" {
  name         = "vscode-gpu"
  machine_type = var.machine_type
  zone         = var.zone

  metadata = {
    enable-oslogin = "TRUE"
  }

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
    }
  }

  network_interface {
    network = "default"

    access_config {
      # Empty block keeps an ephemeral external IP.
    }
  }

  guest_accelerator {
    type  = var.gpu_type
    count = 1
  }

  attached_disk {
    source      = google_compute_disk.perma_disk.id
    device_name = local.perma_disk_name
    mode        = "READ_WRITE"
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
    preemptible         = false
  }
}

resource "google_compute_disk" "perma_disk" {
  name = local.perma_disk_name
  type = "pd-ssd"
  zone = var.zone
  size = var.perma_disk_size_gb

  lifecycle {
    prevent_destroy = true
  }
}
